#include "kernels.h"
#include <cuda_runtime.h>
namespace dlss5::guides::detail {
namespace {
__device__ float rgb_at(const float* p, int y, int x, int c, int h, int w) {
    return p[(max(0,min(h-1,y))*w+max(0,min(w-1,x)))*3+c];
}
__device__ float bilinear_rgb(const float* p, float y, float x, int c, int h, int w) {
    y=fmaxf(y,0.f); x=fmaxf(x,0.f);
    int iy=int(y), ix=int(x); float dy=y-iy, dx=x-ix;
    return (1-dy)*((1-dx)*rgb_at(p,iy,ix,c,h,w)+dx*rgb_at(p,iy,ix+1,c,h,w))+
           dy*((1-dx)*rgb_at(p,iy+1,ix,c,h,w)+dx*rgb_at(p,iy+1,ix+1,c,h,w));
}
__device__ float cubic1(float x) {return ((1.25f*x-2.25f)*x)*x+1;}
__device__ float cubic2(float x) {return ((-.75f*x+3.75f)*x-6)*x+3;}
__device__ void weights(float t, float* v) {
    v[0]=cubic2(t+1); v[1]=cubic1(t); v[2]=cubic1(1-t); v[3]=cubic2(2-t);
}
__global__ void prepare(const float* in,float* out,int h,int w,int rh,int rw,int nh,int nw,bool depth) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=3*nh*nw)return;
    int x=i%nw,y=(i/nw)%nh,c=i/(nh*nw);
    float sx=(float(w)/rw)*(min(x,rw-1)+.5f)-.5f;
    float sy=(float(h)/rh)*(min(y,rh-1)+.5f)-.5f;
    float v;
    if(!depth) v=bilinear_rgb(in,sy,sx,c,h,w)*2-1;
    else {
        int ix=int(floorf(sx)),iy=int(floorf(sy));float cx[4],cy[4];
        weights(sx-ix,cx);weights(sy-iy,cy);v=0;
        for(int yy=0;yy<4;++yy) {
            float row=0;for(int xx=0;xx<4;++xx)row+=cx[xx]*rgb_at(in,iy+yy-1,ix+xx-1,c,h,w);
            v+=cy[yy]*row;
        }
        float mean=c==0?.485f:c==1?.456f:.406f;
        float stdv=c==0?.229f:c==1?.224f:.225f;
        v=(v-mean)/stdv;
    }
    out[i]=v;
}
__device__ float plane_at(const float* p,int y,int x,int c,int rh,int rw,int nh,int nw) {
    return p[c*nh*nw+max(0,min(rh-1,y))*nw+max(0,min(rw-1,x))];
}
__global__ void finish(const float* in,float* out,int h,int w,int rh,int rw,int nh,int nw,bool depth) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;int channels=depth?1:2;if(i>=h*w*channels)return;
    int c=i%channels,x=(i/channels)%w,y=i/(channels*w);
    float sx=depth?(w>1?float(rw-1)/(w-1)*x:0):fmaxf(0,(float(rw)/w)*(x+.5f)-.5f);
    float sy=depth?(h>1?float(rh-1)/(h-1)*y:0):fmaxf(0,(float(rh)/h)*(y+.5f)-.5f);
    int ix=int(sx),iy=int(sy);float dx=sx-ix,dy=sy-iy;
    float v=(1-dy)*((1-dx)*plane_at(in,iy,ix,c,rh,rw,nh,nw)+dx*plane_at(in,iy,ix+1,c,rh,rw,nh,nw))+
             dy*((1-dx)*plane_at(in,iy+1,ix,c,rh,rw,nh,nw)+dx*plane_at(in,iy+1,ix+1,c,rh,rw,nh,nw));
    out[i]=depth?v:v*(c==0?float(w)/rw:float(h)/rh);
}
__global__ void insert(const float* src,float* dst,int spatial,int channels,int slot) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=spatial*channels)return;
    dst[(i/channels*31+slot)*channels+i%channels]=src[i];
}
}
void flow_prepare(const float* in,float* out,int h,int w,int rh,int rw,int nh,int nw,cudaStream_t s) {
    prepare<<<(3*nh*nw+255)/256,256,0,s>>>(in,out,h,w,rh,rw,nh,nw,false);
}
void depth_prepare(const float* in,float* out,int h,int w,int nh,int nw,cudaStream_t s) {
    prepare<<<(3*nh*nw+255)/256,256,0,s>>>(in,out,h,w,nh,nw,nh,nw,true);
}
void flow_finish(const float* in,float* out,int h,int w,int rh,int rw,int nh,int nw,cudaStream_t s) {
    finish<<<(2*h*w+255)/256,256,0,s>>>(in,out,h,w,rh,rw,nh,nw,false);
}
void depth_finish(const float* in,float* out,int h,int w,int nh,int nw,cudaStream_t s) {
    finish<<<(h*w+255)/256,256,0,s>>>(in,out,h,w,nh,nw,nh,nw,true);
}
void cache_insert(const float* in,float* out,int spatial,int channels,int slot,cudaStream_t s) {
    insert<<<(spatial*channels+255)/256,256,0,s>>>(in,out,spatial,channels,slot);
}
}
