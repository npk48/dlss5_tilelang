#define NOMINMAX
#define ORT_API_MANUAL_INIT
#include "guides.h"
#include "kernels.h"
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dlss5::guides {
namespace {
void ck(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string("guides CUDA: ")+cudaGetErrorString(e));
}
struct DeviceScope {
    int old;
    explicit DeviceScope(int device) {ck(cudaGetDevice(&old));ck(cudaSetDevice(device));}
    ~DeviceScope(){cudaSetDevice(old);}
};
struct Tensor {
    float* data=nullptr;
    size_t count=0;
    std::vector<int64_t> shape;
    explicit Tensor(std::vector<int64_t> s):shape(std::move(s)) {
        count=1;
        for(auto n:shape) {
            if(n<=0 || count>std::numeric_limits<size_t>::max()/sizeof(float)/size_t(n))
                throw std::runtime_error("guides: invalid ONNX tensor shape");
            count*=size_t(n);
        }
        ck(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(float)));
    }
    ~Tensor(){if(data)cudaFree(data);}
    Tensor(const Tensor&)=delete;
    Tensor& operator=(const Tensor&)=delete;
    Ort::Value value(const Ort::MemoryInfo& memory) {
        return Ort::Value::CreateTensor<float>(memory,data,count,shape.data(),shape.size());
    }
};
using Frame=std::vector<std::shared_ptr<Tensor>>;
int even_round(double x) {
    double lo=std::floor(x), f=x-lo;
    return int(lo+(f>.5 || (f==.5 && std::fmod(lo,2.)!=0)));
}
std::pair<int,int> depth_hw(int h,int w,int size) {
    double ratio=double(std::max(h,w))/std::min(h,w);
    if(ratio>1.78) size=even_round(int(size*1.777/ratio)/14.)*14;
    if(size<14)throw std::invalid_argument("guides: aspect ratio too extreme for VDA preprocessing");
    double scale=std::max(double(size)/h,double(size)/w);
    auto multiple=[size](double x){int n=even_round(x/14.)*14;return n<size?int(std::ceil(x/14.))*14:n;};
    return {multiple(h*scale),multiple(w*scale)};
}
void pointer(const float* p,int device) {
    if(!p)throw std::invalid_argument("guides: null device buffer");
    cudaPointerAttributes a{};ck(cudaPointerGetAttributes(&a,p));
    if(a.type!=cudaMemoryTypeDevice || a.device!=device)
        throw std::invalid_argument("guides: expected device buffer on configured CUDA device");
}
void initialize_ort() {
    // The C++ wrapper normally calls OrtGetApiBase in a static initializer,
    // defeating /DELAYLOAD and breaking SDK startup even with guides disabled.
    // Hold a process-lifetime module reference before initializing the wrapper.
    static const OrtApi* api=[] {
        HMODULE module=nullptr;
        if(!GetModuleHandleExW(0,L"onnxruntime.dll",&module))
            module=LoadLibraryExW(L"onnxruntime.dll",nullptr,LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!module)throw std::runtime_error("guides unavailable: cannot load onnxruntime.dll (Windows error "+std::to_string(GetLastError())+
            "); deploy native ORT runtime beside the EXE or register its DLL directory");
        using GetBase=const OrtApiBase* (ORT_API_CALL*)();
        auto base=reinterpret_cast<GetBase>(GetProcAddress(module,"OrtGetApiBase"));
        const OrtApi* result=base?base()->GetApi(ORT_API_VERSION):nullptr;
        if(!result){FreeLibrary(module);throw std::runtime_error("guides unavailable: incompatible ONNX Runtime API; require 1.20+");}
        Ort::InitApi(result);
        return result;
    }();
    (void)api;
}
Ort::Env& environment(){static Ort::Env env(ORT_LOGGING_LEVEL_WARNING,"dlss5_guides");return env;}
}

struct Engine::Impl {
    std::filesystem::path directory;
    int device;
    Config config;
    bool stream_set=false, poisoned=false;
    cudaStream_t stream=nullptr;
    uint64_t frames=0;
    int source_h=0,source_w=0,nh=0,nw=0;
    std::unique_ptr<Ort::MemoryInfo> memory;
    std::unique_ptr<Ort::Session> flow_session,init_session,step_session;
    std::unique_ptr<Tensor> current_flow,previous_flow,flow_output,depth_input,depth_output;
    std::vector<std::unique_ptr<Tensor>> packed;
    std::vector<std::shared_ptr<Frame>> history;

    Impl(const std::string& dir,int dev,Config cfg):directory(std::filesystem::u8path(dir)),device(dev),config(cfg),
        memory(nullptr) {
        if(cfg.flow_updates<1 || cfg.flow_longest_side<128 || cfg.depth_input_size<28)
            throw std::invalid_argument("guides: invalid estimator configuration");
        DeviceScope scope(device);
        if(cfg.flow || cfg.depth) {
            if(!std::filesystem::is_directory(directory))
                throw std::runtime_error("guides unavailable: model directory missing: "+dir);
            initialize_ort();
            memory=std::make_unique<Ort::MemoryInfo>("Cuda",OrtDeviceAllocator,dev,OrtMemTypeDefault);
        }
    }
    ~Impl(){DeviceScopeNoThrow();}
    void DeviceScopeNoThrow() noexcept {
        int old=0;cudaGetDevice(&old);cudaSetDevice(device);
        if(stream_set)cudaStreamSynchronize(stream);
        clear();cudaSetDevice(old);
    }
    void clear() {
        history.clear();packed.clear();depth_output.reset();depth_input.reset();
        flow_output.reset();previous_flow.reset();current_flow.reset();
        step_session.reset();init_session.reset();flow_session.reset();
        stream_set=false;stream=nullptr;frames=0;source_h=source_w=nh=nw=0;poisoned=false;
    }
    void begin(int h,int w,CUstream s) {
        if(h<=0 || w<=0 || h>32768 || w>32768 || int64_t(h)*w>std::numeric_limits<int>::max()/4)
            throw std::invalid_argument("guides: invalid image dimensions");
        if(poisoned)throw std::runtime_error("guides: prior inference failed; reset required");
        if(stream_set && stream!=reinterpret_cast<cudaStream_t>(s))
            throw std::invalid_argument("guides: caller stream changed; reset required");
        stream=reinterpret_cast<cudaStream_t>(s);stream_set=true;
        unsigned flags=0;ck(cudaStreamGetFlags(stream,&flags));
    }
    std::unique_ptr<Ort::Session> load(const std::string& file) {
        auto path=directory/file;
        if(!std::filesystem::is_regular_file(path))
            throw std::runtime_error("guides unavailable: missing "+path.u8string()+
                "; export this model/configuration offline with native/tools/export_guides.py");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if(const char* directory=std::getenv("DLSS5_GUIDES_PROFILE_DIR")) {
            auto prefix=std::filesystem::u8path(directory)/file;
            options.EnableProfiling(prefix.c_str());
        }
        // Shape/control scalars may use CPU kernels. Image/cache tensors are
        // bound to CUDA memory; no host image preprocess or output staging.
        OrtCUDAProviderOptionsV2* raw=nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&raw));
        std::unique_ptr<OrtCUDAProviderOptionsV2,void(*)(OrtCUDAProviderOptionsV2*)> cuda_options(
            raw,Ort::GetApi().ReleaseCUDAProviderOptions);
        std::string dev=std::to_string(device);
        std::string user=std::to_string(reinterpret_cast<uintptr_t>(stream));
        const char* keys[]={"device_id","user_compute_stream","do_copy_in_default_stream","use_tf32","cudnn_conv_algo_search","cudnn_conv_use_max_workspace"};
        const char* vals[]={dev.c_str(),user.c_str(),"1","0","HEURISTIC","0"};
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(raw,keys,vals,6));
        options.AppendExecutionProvider_CUDA_V2(*raw);
        try {return std::make_unique<Ort::Session>(environment(),path.c_str(),options);}
        catch(const Ort::Exception& e){throw std::runtime_error("guides unavailable: ONNX Runtime CUDA load failed for "+file+": "+e.what());}
    }
    static void ensure(std::unique_ptr<Tensor>& t,std::vector<int64_t> shape) {
        if(!t || t->shape!=shape)t=std::make_unique<Tensor>(std::move(shape));
    }
    static void bind_input(Ort::IoBinding& io,const char* name,Tensor& t,const Ort::MemoryInfo& m) {
        auto v=t.value(m);io.BindInput(name,v);
    }
    static void bind_output(Ort::IoBinding& io,const char* name,Tensor& t,const Ort::MemoryInfo& m) {
        auto v=t.value(m);io.BindOutput(name,v);
    }
    void run(Ort::Session& session,Ort::IoBinding& io) {
        ck(cudaGetLastError());
        Ort::RunOptions run_options;
        // Default Run synchronization keeps Ort-managed intermediates alive;
        // CUDA pre/postprocessing and all neural operations use caller stream.
        session.Run(run_options,io);
    }
};

Engine::Engine(const std::string& model_dir,int device,Config config):impl_(std::make_unique<Impl>(model_dir,device,config)){}
Engine::~Engine()=default;
uint64_t Engine::depth_frames()const noexcept{return impl_->frames;}
void Engine::reset_history(){DeviceScope scope(impl_->device);if(impl_->stream_set)ck(cudaStreamSynchronize(impl_->stream));impl_->history.clear();impl_->frames=0;impl_->poisoned=false;}
void Engine::reset(){DeviceScope scope(impl_->device);if(impl_->stream_set)ck(cudaStreamSynchronize(impl_->stream));impl_->clear();}

void Engine::estimate_flow(const float* current,const float* previous,int h,int w,float* motion,CUstream stream) {
    auto& p=*impl_;DeviceScope scope(p.device);
    if(!p.config.flow)throw std::runtime_error("guides unavailable: flow component disabled");
    p.begin(h,w,stream);pointer(current,p.device);pointer(previous,p.device);pointer(motion,p.device);
    if(motion==current || motion==previous)throw std::invalid_argument("guides: output aliases input");
    try {
        if(!p.flow_session)p.flow_session=p.load("raft_small_u"+std::to_string(p.config.flow_updates)+".onnx");
        double scale=std::min(1.,double(p.config.flow_longest_side)/std::max(h,w));
        int rh=std::max(1,even_round(h*scale)),rw=std::max(1,even_round(w*scale));
        int nh=std::max(128,(rh+7)/8*8),nw=std::max(128,(rw+7)/8*8);
        Impl::ensure(p.current_flow,{1,3,nh,nw});Impl::ensure(p.previous_flow,{1,3,nh,nw});Impl::ensure(p.flow_output,{1,2,nh,nw});
        detail::flow_prepare(current,p.current_flow->data,h,w,rh,rw,nh,nw,p.stream);
        detail::flow_prepare(previous,p.previous_flow->data,h,w,rh,rw,nh,nw,p.stream);
        Ort::IoBinding io(*p.flow_session);
        Impl::bind_input(io,"current",*p.current_flow,*p.memory);Impl::bind_input(io,"previous",*p.previous_flow,*p.memory);
        Impl::bind_output(io,"flow",*p.flow_output,*p.memory);p.run(*p.flow_session,io);
        detail::flow_finish(p.flow_output->data,motion,h,w,rh,rw,nh,nw,p.stream);ck(cudaGetLastError());
    }catch(...){p.poisoned=true;throw;}
}

void Engine::estimate_depth(const float* current,int h,int w,float* depth,CUstream stream) {
    auto& p=*impl_;DeviceScope scope(p.device);
    if(!p.config.depth)throw std::runtime_error("guides unavailable: depth component disabled");
    p.begin(h,w,stream);pointer(current,p.device);pointer(depth,p.device);
    if(depth==current)throw std::invalid_argument("guides: output aliases input");
    if(p.frames && (p.source_h!=h || p.source_w!=w))throw std::invalid_argument("guides: depth source size changed; reset required");
    try {
        if(!p.init_session) {
            auto shape=depth_hw(h,w,p.config.depth_input_size);p.nh=shape.first;p.nw=shape.second;
            std::string prefix="vda_small_"+std::to_string(p.nh)+"x"+std::to_string(p.nw);
            // Load BOTH before any state advances. A missing step is not usable VDA.
            auto init=p.load(prefix+"_init.onnx"),step=p.load(prefix+"_step.onnx");
            if(init->GetInputCount()!=1 || init->GetOutputCount()!=9 || step->GetInputCount()!=9 || step->GetOutputCount()!=9)
                throw std::runtime_error("guides: VDA graph does not expose eight causal hidden states");
            p.init_session=std::move(init);p.step_session=std::move(step);
            Impl::ensure(p.depth_input,{1,3,p.nh,p.nw});Impl::ensure(p.depth_output,{1,1,p.nh,p.nw});
            for(size_t i=0;i<8;++i) {
                auto shape=p.step_session->GetInputTypeInfo(i+1).GetTensorTypeAndShapeInfo().GetShape();
                if(shape.size()!=3 || shape[1]!=31)throw std::runtime_error("guides: invalid VDA cache input shape");
                p.packed.emplace_back(std::make_unique<Tensor>(shape));
            }
        }
        bool first=p.frames==0;
        auto& session=first?*p.init_session:*p.step_session;
        auto next=std::make_shared<Frame>();
        for(size_t i=0;i<8;++i) {
            auto shape=session.GetOutputTypeInfo(i+1).GetTensorTypeAndShapeInfo().GetShape();
            if(shape.size()!=3 || shape[1]!=1)throw std::runtime_error("guides: invalid VDA new-cache shape");
            next->push_back(std::make_shared<Tensor>(shape));
        }
        Ort::IoBinding io(session);
        detail::depth_prepare(current,p.depth_input->data,h,w,p.nh,p.nw,p.stream);
        Impl::bind_input(io,"rgb",*p.depth_input,*p.memory);
        if(!first) {
            // EXACT reference selection: history[:2] + history[-29:]. Each
            // hidden state is [spatial, time, channel], NOT concatenated planes.
            for(size_t i=0;i<8;++i) {
                auto& packed=*p.packed[i];
                for(size_t slot=0;slot<31;++slot) {
                    size_t index=slot<2?slot:p.history.size()-29+(slot-2);
                    detail::cache_insert((*p.history[index])[i]->data,packed.data,int(packed.shape[0]),int(packed.shape[2]),int(slot),p.stream);
                }
                Impl::bind_input(io,("cache_"+std::to_string(i)).c_str(),packed,*p.memory);
            }
        }
        Impl::bind_output(io,"depth",*p.depth_output,*p.memory);
        for(size_t i=0;i<8;++i)Impl::bind_output(io,("new_cache_"+std::to_string(i)).c_str(),*(*next)[i],*p.memory);
        p.run(session,io);
        detail::depth_finish(p.depth_output->data,depth,h,w,p.nh,p.nw,p.stream);ck(cudaGetLastError());
        if(first)p.history.assign(32,next);else p.history.push_back(next);
        // reference uses zero-based index; preserves first anchor, removes second.
        if(p.frames+32>42)p.history.erase(p.history.begin()+1);
        ++p.frames;p.source_h=h;p.source_w=w;
    }catch(...){p.poisoned=true;throw;}
}
} // namespace dlss5::guides
