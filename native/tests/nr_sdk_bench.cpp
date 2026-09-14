#define D5_STATIC
#include "dlss5.h"
#include <windows.h>
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>
using Clock=std::chrono::steady_clock;
static void ck(CUresult r){if(r)throw std::runtime_error("CUDA "+std::to_string(r));}
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
struct Api {
 HMODULE module;D5Session session=nullptr;D5Frame frame{};CUdeviceptr output=0;double create_ms=0,prepare_ms=0;uint64_t index=0;std::vector<double> all,rounds;
#define DECL(name) decltype(&name) name##_fn
 DECL(d5_default_config);DECL(d5_create);DECL(d5_prepare);DECL(d5_process);DECL(d5_synchronize);DECL(d5_destroy);DECL(d5_last_error);
#undef DECL
 void check(D5Status r){if(r!=D5_OK)throw std::runtime_error(d5_last_error_fn());}
 Api(const char* dll,const char* model,CUdeviceptr input,int w,int h,CUstream stream){
  module=LoadLibraryExW(std::filesystem::absolute(dll).c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);if(!module)throw std::runtime_error("DLL load failed");
#define GET(name) name##_fn=(decltype(name##_fn))GetProcAddress(module,#name);if(!name##_fn)throw std::runtime_error(#name)
  GET(d5_default_config);GET(d5_create);GET(d5_prepare);GET(d5_process);GET(d5_synchronize);GET(d5_destroy);GET(d5_last_error);
#undef GET
  ck(cuMemAlloc(&output,size_t(w)*h*16));D5Config config;d5_default_config_fn(&config);config.model_directory_utf8=model;config.enabled_stages=D5_ENABLE_NR;
  auto t=Clock::now();check(d5_create_fn(&config,&session));create_ms=ms(t);t=Clock::now();check(d5_prepare_fn(session,w,h));prepare_ms=ms(t);
  frame.struct_size=sizeof(frame);frame.cuda_stream=stream;frame.color={sizeof(D5Tensor),D5_F32,D5_HWC,4,uint32_t(w),uint32_t(h),input,uint64_t(w)*16,0};frame.output={sizeof(D5Tensor),D5_F32,D5_HWC,4,uint32_t(w),uint32_t(h),output,uint64_t(w)*16,0};frame.delta_ms=1000.f/60;
 }
 ~Api(){if(session)d5_destroy_fn(session);if(output)cuMemFree(output);if(module)FreeLibrary(module);}
 double run(){frame.frame_id=index;frame.reset=index++==0;auto t=Clock::now();check(d5_process_fn(session,&frame));check(d5_synchronize_fn(session));ck(cuStreamSynchronize((CUstream)frame.cuda_stream));return ms(t);}
 void save(const std::filesystem::path& file,size_t n){std::vector<float> data(n);ck(cuMemcpyDtoH(data.data(),output,n*4));for(auto v:data)if(!std::isfinite(v))throw std::runtime_error("nonfinite image");std::ofstream f(file,std::ios::binary);f.write((char*)data.data(),n*4);}
};
int main(int argc,char**argv){try{
 if(argc!=4&&argc!=5)throw std::runtime_error("DLL MODEL_DIR OUTPUT_DIR [COMPARE_DLL]");std::filesystem::path out=argv[3];std::filesystem::create_directories(out);
 ck(cuInit(0));CUdevice dev;CUcontext context;CUstream stream;ck(cuDeviceGet(&dev,0));ck(cuDevicePrimaryCtxRetain(&context,dev));ck(cuCtxSetCurrent(context));ck(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
 constexpr int w=320,h=288;size_t n=size_t(w)*h*4;std::vector<float> image(n);for(int y=0;y<h;y++)for(int x=0;x<w;x++){size_t i=(size_t(y)*w+x)*4;image[i]=.5f+.2f*std::sin(x*.071f);image[i+1]=.5f+.25f*std::cos(y*.091f);image[i+2]=.5f+.2f*std::sin(x*.13f+y*.09f);image[i+3]=1;}
 CUdeviceptr input;ck(cuMemAlloc(&input,n*4));ck(cuMemcpyHtoD(input,image.data(),n*4));{
 std::vector<std::unique_ptr<Api>> apis;apis.push_back(std::make_unique<Api>(argv[1],argv[2],input,w,h,stream));if(argc==5)apis.push_back(std::make_unique<Api>(argv[4],argv[2],input,w,h,stream));
 for(int i=0;i<2;i++)for(size_t a=0;a<apis.size();a++){apis[a]->run();apis[a]->save(out/("api"+std::to_string(a)+"_image"+std::to_string(i)+".f32"),n);}
 for(int i=0;i<50;i++)for(auto& api:apis)api->run();
 for(int round=0;round<30;round++){for(size_t k=0;k<apis.size();k++){size_t a=round%2?apis.size()-1-k:k;auto& api=*apis[a];api.run();api.run();std::vector<double> batch;for(int j=0;j<10;j++){double value=api.run();batch.push_back(value);api.all.push_back(value);}api.rounds.push_back(median(batch));}}
 std::ofstream report(out/"sdk.json");report<<"{\"w\":"<<w<<",\"h\":"<<h<<",\"iterations_per_api\":300,\"alternating_rounds\":30,\"apis\":[";
 for(size_t a=0;a<apis.size();a++){auto& api=*apis[a];if(a)report<<",";report<<"{\"create_ms\":"<<api.create_ms<<",\"prepare_ms\":"<<api.prepare_ms<<",\"p50_ms\":"<<median(api.all)<<",\"rounds\":[";for(size_t i=0;i<api.rounds.size();i++){if(i)report<<",";report<<api.rounds[i];}report<<"]}";api.save(out/("api"+std::to_string(a)+"_final.f32"),n);std::cout<<"API"<<a<<" prepare_ms="<<api.prepare_ms<<" image_pipeline_p50_ms="<<median(api.all)<<std::endl;}
 report<<"]}";
 }cuMemFree(input);cuStreamDestroy(stream);cuDevicePrimaryCtxRelease(dev);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
