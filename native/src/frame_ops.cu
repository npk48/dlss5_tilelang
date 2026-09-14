#include "frame_ops.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace dlss5::frame_ops {
__device__ float get(D5Tensor a,int x,int y,int c){x=max(0,min(x,int(a.width)-1));y=max(0,min(y,int(a.height)-1));auto pitch=a.row_stride_bytes?a.row_stride_bytes:uint64_t(a.width)*a.channels*4;return ((float*)((char*)a.data+size_t(y)*pitch))[x*a.channels+c];}
__device__ void set(D5Tensor a,int x,int y,int c,float value){auto pitch=a.row_stride_bytes?a.row_stride_bytes:uint64_t(a.width)*a.channels*4;((float*)((char*)a.data+size_t(y)*pitch))[x*a.channels+c]=value;}
__global__ void alpha_k(D5Tensor source,D5Tensor encoded,D5Tensor alpha){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=source.width*source.height)return;int x=i%source.width,y=i/source.width;float a=source.channels==4?get(source,x,y,3):1.f;set(alpha,x,y,0,a);if(source.channels==4)for(int c=0;c<3;++c)set(encoded,x,y,c,get(encoded,x,y,c)*a);}
__global__ void input_k(D5Tensor lin,D5Tensor alpha,D5Tensor prev,bool reset,D5Tensor invalid,D5Tensor reactive,D5Tensor out,D5Tensor combined,float white){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=lin.width*lin.height)return;int x=i%lin.width,y=i/lin.width;float a=get(alpha,x,y,0),r=invalid.data?get(invalid,x,y,0):0.f;r=fmaxf(r,1.f-a);if(!reset)r=fmaxf(r,fabsf(a-get(prev,x,y,0)));if(reactive.data)r=fmaxf(r,get(reactive,x,y,0));set(combined,x,y,0,r);for(int c=0;c<3;++c)set(out,x,y,c,fmaxf(get(lin,x,y,c),0.f)/white);}
__global__ void restore_k(D5Tensor source,D5Tensor out,float white){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=out.width*out.height)return;int x=i%out.width,y=i/out.width;float sx=(x+.5f)*source.width/out.width-.5f,sy=(y+.5f)*source.height/out.height-.5f;int ix=floorf(sx),iy=floorf(sy);float tx=sx-ix,ty=sy-iy;for(int c=0;c<3;++c){float a=fminf(get(source,ix,iy,c),0.f),b=fminf(get(source,ix+1,iy,c),0.f),d=fminf(get(source,ix,iy+1,c),0.f),e=fminf(get(source,ix+1,iy+1,c),0.f);float neg=(a+(b-a)*tx)*(1-ty)+(d+(e-d)*tx)*ty;set(out,x,y,c,get(out,x,y,c)*white+neg);}}
void check(){auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void alpha(const D5Tensor&o,D5Tensor e,D5Tensor a,CUstream stream){alpha_k<<<(o.width*o.height+255)/256,256,0,(cudaStream_t)stream>>>(o,e,a);check();}
void fsr_input(D5Tensor l,D5Tensor a,D5Tensor p,bool reset,D5Tensor i,D5Tensor r,D5Tensor t,D5Tensor c,float white,CUstream stream){input_k<<<(l.width*l.height+255)/256,256,0,(cudaStream_t)stream>>>(l,a,p,reset,i,r,t,c,white);check();}
void fsr_restore(D5Tensor s,D5Tensor t,float w,CUstream stream){restore_k<<<(t.width*t.height+255)/256,256,0,(cudaStream_t)stream>>>(s,t,w);check();}
}
