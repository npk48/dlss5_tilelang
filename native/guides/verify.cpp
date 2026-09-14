#include "guides.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <functional>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

using dlss5::guides::Engine;
using dlss5::guides::Config;
namespace fs=std::filesystem;
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {
    float* p=nullptr;
    explicit Buffer(size_t count){ck(cudaMalloc(reinterpret_cast<void**>(&p),count*sizeof(float)));}
    ~Buffer(){cudaFree(p);}
};
std::vector<float> read(const fs::path& path,size_t count) {
    std::vector<float> v(count);std::ifstream f(path,std::ios::binary);
    if(!f.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*sizeof(float))))throw std::runtime_error("Cannot read "+path.string());
    return v;
}
std::vector<float> download(float* device,size_t n,cudaStream_t stream) {
    std::vector<float> v(n);ck(cudaMemcpyAsync(v.data(),device,n*sizeof(float),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));
    for(float x:v)if(!std::isfinite(x))throw std::runtime_error("nonfinite native output");
    return v;
}
void save(const fs::path& path,const std::vector<float>& v) {
    std::ofstream f(path,std::ios::binary);if(!f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*sizeof(float))))throw std::runtime_error("Cannot write result");
}
void must_throw(const char* label,const std::function<void()>& fn) {
    try {fn();}catch(const std::exception& e){std::cout<<"EXPECTED "<<label<<": "<<e.what()<<"\n";return;}
    throw std::runtime_error(std::string("Expected rejection: ")+label);
}
void same(const std::vector<float>& a,const std::vector<float>& b,const char* label) {
    float error=0;for(size_t i=0;i<a.size();++i)error=std::max(error,std::abs(a[i]-b[i]));
    if(error>1e-4f)throw std::runtime_error(std::string(label)+" mismatch "+std::to_string(error));
    std::cout<<label<<" max_abs="<<error<<"\n";
}
int main(int argc,char** argv) {
    if(argc==3 && std::string(argv[1])=="--missing-backend") {
        try {
            Config disabled;disabled.flow=disabled.depth=false;
            Engine off(argv[2],0,disabled);
            try {Engine enabled(argv[2],0);}
            catch(const std::runtime_error& e) {
                if(std::string(e.what()).find("cannot load onnxruntime.dll")==std::string::npos &&
                   std::string(e.what()).find("incompatible ONNX Runtime API")==std::string::npos)throw;
                std::cout<<"PASS disabled backend needs no ORT; enabled throws: "<<e.what()<<"\n";return 0;
            }
            throw std::runtime_error("Missing-backend test unexpectedly loaded ORT");
        }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
    }
    if(argc!=10){std::cerr<<"Usage: guides_verify models fixtures h w frames updates flow_limit depth_size all|flow|depth\n";return 2;}
    try {
        fs::path models=argv[1],work=argv[2];int h=std::stoi(argv[3]),w=std::stoi(argv[4]),frames=std::stoi(argv[5]);
        if(h<=0 || w<=0 || frames<2)throw std::invalid_argument("Invalid fixture dimensions/frame count");
        Config config;config.flow_updates=std::stoi(argv[6]);config.flow_longest_side=std::stoi(argv[7]);config.depth_input_size=std::stoi(argv[8]);
        // argv[9] is component; argc includes executable (10 total).
        std::string component=argv[9];config.flow=component!="depth";config.depth=component!="flow";
        size_t pixels=size_t(h)*w;
        ck(cudaSetDevice(0));cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        {
            Buffer current(pixels*3),previous(pixels*3),motion(pixels*2),depth(pixels);
            Engine engine(models.u8string(),0,config);
            std::vector<float> first_depth;
            for(int i=0;i<frames;++i) {
                auto image=read(work/("rgb_"+std::to_string(i)+".f32"),pixels*3);
                ck(cudaMemcpyAsync(current.p,image.data(),pixels*3*sizeof(float),cudaMemcpyHostToDevice,stream));
                if(config.flow && i) {
                    engine.estimate_flow(current.p,previous.p,h,w,motion.p,reinterpret_cast<CUstream>(stream));
                    save(work/("flow_native_"+std::to_string(i)+".f32"),download(motion.p,pixels*2,stream));
                }
                if(config.depth) {
                    engine.estimate_depth(current.p,h,w,depth.p,reinterpret_cast<CUstream>(stream));
                    auto values=download(depth.p,pixels,stream);
                    if(i==0)first_depth=values;
                    save(work/("depth_native_"+std::to_string(i)+".f32"),values);
                    if(engine.depth_frames()!=uint64_t(i+1))throw std::runtime_error("incorrect depth frame counter");
                }
                ck(cudaMemcpyAsync(previous.p,current.p,pixels*3*sizeof(float),cudaMemcpyDeviceToDevice,stream));
                ck(cudaStreamSynchronize(stream));std::cout<<"NATIVE frame "<<i<<"\n";
            }
            if(config.depth) {
                must_throw("source size change",[&]{engine.estimate_depth(current.p,h-1,w,depth.p,reinterpret_cast<CUstream>(stream));});
                auto image=read(work/"rgb_0.f32",pixels*3);
                ck(cudaMemcpyAsync(current.p,image.data(),pixels*3*sizeof(float),cudaMemcpyHostToDevice,stream));
                {
                    Config dc=config;dc.flow=false;Engine independent(models.u8string(),0,dc);
                    independent.estimate_depth(current.p,h,w,depth.p,reinterpret_cast<CUstream>(stream));
                    same(first_depth,download(depth.p,pixels,stream),"independent_state");
                    if(engine.depth_frames()!=uint64_t(frames))throw std::runtime_error("cross-engine state mutation");
                }
                engine.reset();
                if(engine.depth_frames()!=0)throw std::runtime_error("reset frame counter");
                engine.estimate_depth(current.p,h,w,depth.p,reinterpret_cast<CUstream>(stream));
                same(first_depth,download(depth.p,pixels,stream),"reset_bootstrap");
            }
            Config disabled=config;disabled.flow=disabled.depth=false;Engine off(models.u8string(),0,disabled);
            must_throw("disabled flow",[&]{off.estimate_flow(current.p,previous.p,h,w,motion.p,reinterpret_cast<CUstream>(stream));});
            must_throw("disabled depth",[&]{off.estimate_depth(current.p,h,w,depth.p,reinterpret_cast<CUstream>(stream));});
            fs::path empty=work/"empty_models";fs::create_directories(empty);
            Engine unavailable(empty.u8string(),0,config);
            if(config.flow)must_throw("missing RAFT asset",[&]{unavailable.estimate_flow(current.p,previous.p,h,w,motion.p,reinterpret_cast<CUstream>(stream));});
            unavailable.reset();
            if(config.depth)must_throw("missing VDA asset",[&]{unavailable.estimate_depth(current.p,h,w,depth.p,reinterpret_cast<CUstream>(stream));});
            engine.reset();
        }
        ck(cudaStreamDestroy(stream));
#ifdef _WIN32
        for(const wchar_t* name:{L"python311.dll",L"torch_cpu.dll",L"torch_cuda.dll"})
            if(GetModuleHandleW(name))throw std::runtime_error("Unexpected Python/Torch runtime dependency");
        for(const wchar_t* name:{L"onnxruntime.dll",L"onnxruntime_providers_cuda.dll",L"cudnn64_9.dll",L"cublas64_12.dll",L"cufft64_11.dll"}) {
            HMODULE module=GetModuleHandleW(name);if(!module)throw std::runtime_error("Expected native GPU backend DLL not loaded");
            wchar_t path[32768];GetModuleFileNameW(module,path,32768);std::wcout<<L"LOADED "<<path<<L"\n";
        }
#endif
        std::cout<<"PASS native stream/state/error checks\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
