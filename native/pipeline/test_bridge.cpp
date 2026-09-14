#include "pipeline.h"
#include <cuda_runtime_api.h>
#include <exception>
#include <stdexcept>
#include <string>
using namespace dlss5::pipeline;
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
static thread_local std::string error;
#define TRY try {
#define CATCH return 0; } catch(const std::exception& e) {error=e.what();return 1;}
EXPORT const char* pipeline_error(){return error.c_str();}
EXPORT void* pipeline_create(unsigned w,unsigned h){try{auto* p=new PipelineState;p->allocate(w,h);return p;}catch(const std::exception& e){error=e.what();return nullptr;}}
EXPORT void pipeline_destroy(void* p){delete static_cast<PipelineState*>(p);}
EXPORT int pipeline_prepare(void* p,D5Tensor* work,D5Tensor* motion,D5Tensor* conf,D5NRSettings* settings,int reset,void* stream){TRY static_cast<PipelineState*>(p)->prepare(*work,*motion,*conf,*settings,reset,reinterpret_cast<CUstream>(stream));CATCH}
EXPORT int pipeline_view(void* p,int which,D5Tensor* result){TRY auto& s=*static_cast<PipelineState*>(p);switch(which){case 0:*result=s.packet();break;case 1:*result=s.current();break;case 2:*result=s.sampled_current();break;case 3:*result=s.warped_history();break;case 4:*result=s.gate();break;case 5:*result=s.padded_output();break;case 6:*result=s.history();break;case 7:*result=s.output();break;}CATCH}
EXPORT int pipeline_composite(void* p,D5Tensor* head,void* stream){TRY static_cast<PipelineState*>(p)->composite(*head,reinterpret_cast<CUstream>(stream));CATCH}
EXPORT int pipeline_commit(void* p){TRY static_cast<PipelineState*>(p)->commit();CATCH}
EXPORT int pipeline_resize(D5Tensor* src,D5Tensor* dst,int aa,void* stream){TRY Ops::resize(*src,*dst,reinterpret_cast<CUstream>(stream),aa!=0);CATCH}
EXPORT int pipeline_color(D5Tensor* src,D5Tensor* dst,ColorOptions* c,int encode,void* stream){TRY if(encode)Ops::encode(*src,*dst,*c,reinterpret_cast<CUstream>(stream));else Ops::preprocess(*src,*dst,*c,reinterpret_cast<CUstream>(stream));CATCH}
EXPORT int pipeline_finish(D5Tensor* ctx,D5Tensor* ref,D5Tensor* mod,D5Tensor* original,D5Tensor* protect,D5Tensor* out,ColorOptions* c,float mix,void* stream){TRY Ops::finish(*ctx,*ref,*mod,*original,*protect,*out,*c,mix,reinterpret_cast<CUstream>(stream));CATCH}
EXPORT int pipeline_transport(D5Tensor* ref,D5Tensor* result,D5Tensor* base,D5Tensor* out,void* stream){TRY Ops::transport(*ref,*result,*base,*out,reinterpret_cast<CUstream>(stream));CATCH}
EXPORT int pipeline_copy(D5Tensor* src,D5Tensor* dst,void* stream){TRY size_t row=src->width*(src->layout==D5_CHW?1:src->channels)*4;for(unsigned k=0;k<(src->layout==D5_CHW?src->channels:1);k++){auto e=cudaMemcpy2DAsync(reinterpret_cast<void*>(dst->data+k*row*src->height),row,reinterpret_cast<void*>(src->data+k*row*src->height),src->row_stride_bytes?src->row_stride_bytes:row,row,src->height,cudaMemcpyDeviceToDevice,reinterpret_cast<CUstream>(stream));if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}CATCH}
EXPORT void* pipeline_guide_create(unsigned w,unsigned h){try{auto* p=new GuideState;p->allocate(w,h);return p;}catch(const std::exception& e){error=e.what();return nullptr;}}
EXPORT void pipeline_guide_destroy(void* p){delete static_cast<GuideState*>(p);}
EXPORT int pipeline_guide_prepare(void* p,D5Tensor* color,D5Tensor* motion,D5Tensor* depth,int reset,int luma,void* stream){TRY GuideOptions c;c.luma=luma!=0;static_cast<GuideState*>(p)->prepare(*color,*motion,*depth,reset,reinterpret_cast<CUstream>(stream),c);CATCH}
EXPORT int pipeline_guide_view(void* p,int which,D5Tensor* result){TRY auto& s=*static_cast<GuideState*>(p);*result=which==0?s.motion():which==1?s.distrust():s.tests();CATCH}
EXPORT int pipeline_guide_commit(void* p){TRY static_cast<GuideState*>(p)->commit();CATCH}
EXPORT int pipeline_reconstruct(D5Tensor* src,D5Tensor* dst,int mode,void* stream){TRY Ops::reconstruct(*src,*dst,static_cast<Reconstruction>(mode),reinterpret_cast<CUstream>(stream));CATCH}
