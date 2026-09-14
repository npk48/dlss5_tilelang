// Paired static/dynamic NR evaluator. Links unchanged baseline or current Engine.
#include "nr_engine.h"
#include <cuda.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <cstring>
using Clock=std::chrono::steady_clock;
static void ck(CUresult r){if(r){const char* s=nullptr;cuGetErrorString(r,&s);throw std::runtime_error(s?s:"CUDA error");}}
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Buffer{CUdeviceptr p=0;Buffer(size_t n){ck(cuMemAlloc(&p,n));}~Buffer(){if(p)cuMemFree(p);}};
static std::vector<float> load(const std::filesystem::path& p,size_t n){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(n*sizeof(float)))throw std::runtime_error("bad packet: "+p.string());std::vector<float> v(n);f.seekg(0);f.read((char*)v.data(),n*sizeof(float));return v;}
static double percentile(std::vector<double> v,double q){std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(q*(v.size()-1)))];}
int main(int argc,char**argv){try{
 if(argc<8||(argc-6)%2)throw std::runtime_error("MODEL_DIR PACKETS_DIR OUTPUT_DIR WARMUP ITERATIONS H W [H W ...]");
 std::filesystem::path model=argv[1],packets=argv[2],out=argv[3];std::filesystem::create_directories(out);
 int warm=std::stoi(argv[4]),count=std::stoi(argv[5]);if(warm<0||count<10)throw std::runtime_error("invalid iteration count");
 ck(cuInit(0));CUdevice dev;CUcontext context;CUstream stream;ck(cuDeviceGet(&dev,0));ck(cuDevicePrimaryCtxRetain(&context,dev));ck(cuCtxSetCurrent(context));ck(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
 auto start=Clock::now();{
 dlss5::nr::Engine engine(model,0);double ctor=ms(start);
 std::ofstream report(out/"timings.jsonl");report<<std::setprecision(12);
 unsigned long long first_compiles=0;
 for(int a=6;a<argc;a+=2){int h=std::stoi(argv[a]),w=std::stoi(argv[a+1]);const std::string key=std::to_string(h)+"x"+std::to_string(w);size_t inCount=size_t(h)*w*16,outCount=size_t(h)*w*4;
  auto x=load(packets/(key+"_0.packet.f32"),inCount),y=load(packets/(key+"_1.packet.f32"),inCount);Buffer first(inCount*4),second(inCount*4),output(outCount*4);
  ck(cuMemcpyHtoD(first.p,x.data(),inCount*4));ck(cuMemcpyHtoD(second.p,y.data(),inCount*4));
  start=Clock::now();engine.prepare(h,w);double prep=ms(start);
  std::vector<float> head(outCount);for(int v=0;v<2;v++){engine.infer(v?second.p:first.p,output.p,h,w,stream);ck(cuMemcpyDtoH(head.data(),output.p,outCount*4));for(float f:head)if(!std::isfinite(f))throw std::runtime_error("nonfinite head");std::ofstream f(out/(key+"_"+std::to_string(v)+".head.f32"),std::ios::binary);f.write((char*)head.data(),outCount*4);}
  for(int i=0;i<warm;i++)engine.infer(i%2?second.p:first.p,output.p,h,w,stream);
  CUevent begin,end;ck(cuEventCreate(&begin,0));ck(cuEventCreate(&end,0));std::vector<double> host,gpu;
  for(int i=0;i<count;i++){ck(cuEventRecord(begin,stream));start=Clock::now();engine.infer(i%2?second.p:first.p,output.p,h,w,stream);host.push_back(ms(start));ck(cuEventRecord(end,stream));ck(cuEventSynchronize(end));float elapsed;ck(cuEventElapsedTime(&elapsed,begin,end));gpu.push_back(elapsed);}
  cuEventDestroy(begin);cuEventDestroy(end);
  report<<"{\"h\":"<<h<<",\"w\":"<<w<<",\"ctor_ms\":"<<ctor<<",\"prepare_ms\":"<<prep<<",\"host_p50_ms\":"<<percentile(host,.5)<<",\"host_p90_ms\":"<<percentile(host,.9)<<",\"gpu_span_p50_ms\":"<<percentile(gpu,.5)<<",\"gpu_span_p90_ms\":"<<percentile(gpu,.9)<<",\"iterations\":"<<count;
#ifdef D5_NR_AUDIT
  const auto stats=engine.audit();if(a==6)first_compiles=stats.nvrtc_compiles;
  if(stats.nvrtc_compiles!=first_compiles||stats.module_loads!=8||stats.nvrtc_compiles+stats.kernel_pack_loads!=8)throw std::runtime_error("NR recompiled/reloaded for runtime geometry");
  report<<",\"nvrtc_compiles\":"<<stats.nvrtc_compiles<<",\"module_loads\":"<<stats.module_loads<<",\"prepares\":"<<stats.prepares<<",\"kernel_pack_loads\":"<<stats.kernel_pack_loads<<",\"compile_ms\":"<<stats.compile_ms<<",\"module_load_ms\":"<<stats.module_load_ms<<",\"upload_ms\":"<<stats.upload_ms<<",\"last_layout_allocation_ms\":"<<stats.last_layout_allocation_ms;
#endif
  report<<"}\n";report.flush();std::cout<<key<<" prepare_ms="<<prep<<" host_p50_ms="<<percentile(host,.5)<<" gpu_span_p50_ms="<<percentile(gpu,.5)<<std::endl;
 }

#ifdef D5_NR_AUDIT
 // Separate engines must not share mutable module constants. Also change the
 // caller stream sequentially: Engine::infer synchronizes before returning.
 dlss5::nr::Engine other(model,0);engine.prepare(320,384);other.prepare(384,320);
 const size_t n=320u*384;auto pa=load(packets/"320x384_0.packet.f32",n*16),pb=load(packets/"384x320_0.packet.f32",n*16);
 Buffer ia(n*64),ib(n*64),oa(n*16),ob(n*16);ck(cuMemcpyHtoD(ia.p,pa.data(),n*64));ck(cuMemcpyHtoD(ib.p,pb.data(),n*64));
 CUstream secondStream;ck(cuStreamCreate(&secondStream,CU_STREAM_NON_BLOCKING));
 std::vector<float> refa(n*4),refb(n*4),actual(n*4);
 engine.infer(ia.p,oa.p,320,384,stream);other.infer(ib.p,ob.p,384,320,secondStream);ck(cuMemcpyDtoH(refa.data(),oa.p,n*16));ck(cuMemcpyDtoH(refb.data(),ob.p,n*16));
 std::exception_ptr errA,errB;
 std::thread ta([&]{try{ck(cuCtxSetCurrent(context));for(int i=0;i<3;i++)engine.infer(ia.p,oa.p,320,384,stream);}catch(...){errA=std::current_exception();}});
 std::thread tb([&]{try{ck(cuCtxSetCurrent(context));for(int i=0;i<3;i++)other.infer(ib.p,ob.p,384,320,secondStream);}catch(...){errB=std::current_exception();}});
 ta.join();tb.join();if(errA)std::rethrow_exception(errA);if(errB)std::rethrow_exception(errB);
 ck(cuMemcpyDtoH(actual.data(),oa.p,n*16));if(std::memcmp(actual.data(),refa.data(),n*16))throw std::runtime_error("Cross-engine geometry corruption A");
 ck(cuMemcpyDtoH(actual.data(),ob.p,n*16));if(std::memcmp(actual.data(),refb.data(),n*16))throw std::runtime_error("Cross-engine geometry corruption B");
 engine.infer(ia.p,oa.p,320,384,secondStream);ck(cuMemcpyDtoH(actual.data(),oa.p,n*16));if(std::memcmp(actual.data(),refa.data(),n*16))throw std::runtime_error("Changed caller stream corrupted geometry");
 ck(cuStreamDestroy(secondStream));std::ofstream(out/"engine-isolation.json")<<"{\"concurrent_engines_bit_exact\":true,\"changed_stream_bit_exact\":true}\n";
 std::cout<<"engines_and_streams PASS"<<std::endl;
#endif
 }cuStreamDestroy(stream);cuDevicePrimaryCtxRelease(dev);return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
