#include "dlss5.h"
#include "nr_engine.h"
#include <cuda.h>
#include <fstream>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <string>
void ck(D5Status s){if(s)throw std::runtime_error(d5_last_error());}
void cu(CUresult s){if(s)throw std::runtime_error("CUDA failure "+std::to_string(s));}
int main(int argc,char**argv){try{if(argc<3)throw std::runtime_error("interface_bench MODEL_DIR PACKET_F32");constexpr int H=320,W=384;std::vector<float>packet(H*W*16);if(!std::ifstream(argv[2],std::ios::binary).read((char*)packet.data(),packet.size()*4))throw std::runtime_error("Cannot read packet");cu(cuInit(0));CUdevice dev;CUcontext ctx;cu(cuDeviceGet(&dev,0));cu(cuDevicePrimaryCtxRetain(&ctx,dev));cu(cuCtxPushCurrent(ctx));CUstream stream;cu(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));CUdeviceptr p,a,b;cu(cuMemAlloc(&p,packet.size()*4));cu(cuMemAlloc(&a,H*W*16));cu(cuMemAlloc(&b,H*W*16));cu(cuMemcpyHtoD(p,packet.data(),packet.size()*4));
 D5Config cfg;d5_default_config(&cfg);cfg.model_directory_utf8=argv[1];D5Session sdk;ck(d5_create(&cfg,&sdk));ck(d5_prepare_packet(sdk,W,H));dlss5::nr::Engine direct(std::filesystem::u8path(argv[1]),0);direct.prepare(H,W);D5Tensor input{sizeof(D5Tensor),D5_F32,D5_CHW,16,W,H,p,W*4,W*H*4};D5Tensor output{sizeof(D5Tensor),D5_F32,D5_HWC,4,W,H,b,W*16,0};
 std::vector<double>d,c;for(int i=0;i<24;++i){for(int side=0;side<2;++side){bool api=((i+side)&1)!=0;auto start=std::chrono::steady_clock::now();if(api)ck(d5_infer_packet(sdk,&input,&output,stream));else direct.infer(p,a,H,W,stream);double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();if(i>=4)(api?c:d).push_back(ms);}}
 std::vector<float>x(H*W*4),y(x.size());cu(cuMemcpyDtoH(x.data(),a,x.size()*4));cu(cuMemcpyDtoH(y.data(),b,y.size()*4));if(x!=y)throw std::runtime_error("ABI/direct mismatch");std::sort(c.begin(),c.end());std::sort(d.begin(),d.end());auto cm=c[c.size()/2],dm=d[d.size()/2];std::cout<<"{\"direct_median_ms\":"<<dm<<",\"dll_median_ms\":"<<cm<<",\"difference_ms\":"<<cm-dm<<",\"samples_each\":"<<c.size()<<",\"bit_exact\":true}\n";ck(d5_destroy(sdk));cu(cuMemFree(p));cu(cuMemFree(a));cu(cuMemFree(b));cu(cuStreamDestroy(stream));return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
