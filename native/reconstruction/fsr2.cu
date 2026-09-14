// FSR2 v2.2.1 equations ported from reference/whitebox_pipeline/fsr2.py.
// AMD reference algorithm copyright 2022-2023 AMD, MIT (FSR2_LICENSE.txt).
#include "fsr2.h"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
namespace dlss5::reconstruction {
namespace {
void ck(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
__device__ float sat(float x) { return fminf(fmaxf(x, 0), 1); }
__device__ float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ float halfq(float x) { return __half2float(__float2half_rz(x)); }
__device__ float uq(float x) { return nearbyintf(sat(x) * 255) * (1.f / 255); }
__device__ bool valid(int x, int y, int w, int h) {
  return x >= 0 && y >= 0 && x < w && y < h;
}
__device__ float ld(const float *a, int x, int y, int w, int h, int c = 0,
                    int n = 1, bool clamp = false) {
  if (!a)
    return 0;
  if (!clamp && !valid(x, y, w, h))
    return 0;
  return a[(min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)) * n + c];
}
__device__ float sample(const float *a, float u, float v, int w, int h,
                        int c = 0, int n = 1) {
  // Match the reference grid_sample UV -> [-1,1] -> pixel boundary and
  // weighted CUDA sampler. Its rounding affects discrete luma oscillation tests.
  float gx=u*2-1,gy=v*2-1;
  float x=fminf(fmaxf(fmaf(gx+1,float(w),-1)*.5f,0),float(w-1));
  float y=fminf(fmaxf(fmaf(gy+1,float(h),-1)*.5f,0),float(h-1));
  int ix=floorf(x),iy=floorf(y);
  float nw=(ix+1-x)*(iy+1-y),ne=(x-ix)*(iy+1-y);
  float sw=(ix+1-x)*(y-iy),se=(x-ix)*(y-iy);
  float result=ld(a,ix,iy,w,h,c,n)*nw;
  result=fmaf(ld(a,ix+1,iy,w,h,c,n),ne,result);
  result=fmaf(ld(a,ix,iy+1,w,h,c,n),sw,result);
  return fmaf(ld(a,ix+1,iy+1,w,h,c,n),se,result);
}
__device__ float length(float x, float y) { return sqrtf(x * x + y * y); }
__device__ float z(float d, float near, float far) {
  float q = far / (near - far);
  return q * near / (d + q);
}
__device__ void yc(float *r) {
  float a = r[0], b = r[1], c = r[2];
  r[0] = .25f * a + .5f * b + .25f * c;
  r[1] = .5f * a - .5f * c;
  r[2] = -.25f * a + .5f * b - .25f * c;
}
__device__ void rgb(float *r) {
  float a = r[0], b = r[1], c = r[2];
  r[0] = a + b - c;
  r[1] = a + c;
  r[2] = a - b - c;
}
__device__ float ratio(float a, float b) {
  float m = fmaxf(a, b);
  return m != 0 ? fminf(a, b) / m : 0;
}
__global__ void input(const float *c, const float *d, const float *m, float *C,
                      float *D, float *M, float *rec, int w, int h, Settings s,
                      bool first) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  for (int k = 0; k < 3; k++) {
    float v = c[i * 3 + k];
    C[i * 3 + k] =
        s.encoding == 0
            ? (v <= .04045f ? v / 12.92f : powf((v + .055f) / 1.055f, 2.4f))
            : v;
  }
  float q = s.camera_far / (s.camera_near - s.camera_far);
  D[i] = sat(-q + q * s.camera_near /
                      fminf(fmaxf(d[i], s.camera_near), s.camera_far));
  M[2 * i] = m[2 * i] / w;
  M[2 * i + 1] = m[2 * i + 1] / h;
  rec[i] = first ? 0 : 1;
}
__global__ void reconstruct(const float *C, const float *D, const float *M,
                            float *dd, float *dm, float *rec, float *lum, int w,
                            int h, int ow, int oh) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w, cx = x, cy = y;
  float d = D[i];
  int xx[8] = {1, 0, 0, -1, -1, 1, -1, 1}, yy[8] = {0, 1, -1, 0, 1, 1, -1, -1};
  for (int k = 0; k < 8; k++) {
    int a = x + xx[k], b = y + yy[k];
    float t = ld(D, a, b, w, h);
    if (valid(a, b, w, h) && t < d) {
      d = t;
      cx = a;
      cy = b;
    }
  }
  float vx = ld(M, cx, cy, w, h, 0, 2), vy = ld(M, cx, cy, w, h, 1, 2);
  dd[i] = d;
  dm[2 * i] = halfq(vx);
  dm[2 * i + 1] = halfq(vy);
  bool moving = length(vx * ow, vy * oh) > .1f;
  float px = ((x + .5f) / w + vx * moving) * w - .5f,
        py = ((y + .5f) / h + vy * moving) * h - .5f;
  int bx = floorf(px), by = floorf(py);
  float fx = px - bx, fy = py - by;
  for (int k = 0; k < 4; k++) {
    int a = bx + k % 2, b = by + k / 2;
    float wt = (k % 2 ? fx : 1 - fx) * (k / 2 ? fy : 1 - fy);
    if (valid(a, b, w, h) && wt > .01f)
      atomicMin((unsigned *)(rec + b * w + a), __float_as_uint(d));
  }
  float l = .2126f * fmaxf(C[3 * i], 0) + .7152f * fmaxf(C[3 * i + 1], 0) +
            .0722f * fmaxf(C[3 * i + 2], 0);
  float perceived =
      (l <= 216.f / 24389 ? l * (24389.f / 27) : powf(l, 1.f / 3) * 116 - 16) *
      .01f;
  lum[i] = halfq(powf(perceived, 1.f / 6));
}
__global__ void mip(const float *C, float *L, int w, int h, int mw, int mh,
                    float jx, float jy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= mw * mh)
    return; // explicit 32x32 SPD tree, preserving addition order
  float a[1024];
  int ox = (i % mw) * 32, oy = (i / mw) * 32;
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++) {
      float u = (ox + x + .5f + jx) / w, v = (oy + y + .5f + jy) / h;
      float l = .2126f * sample(C, u, v, w, h, 0, 3) +
                .7152f * sample(C, u, v, w, h, 1, 3) +
                .0722f * sample(C, u, v, w, h, 2, 3);
      a[y * 32 + x] = valid(ox + x, oy + y, w, h) ? logf(fmaxf(l, .001f)) : 0;
    }
  for (int n = 32; n > 1; n /= 2)
    for (int y = 0; y < n / 2; y++)
      for (int x = 0; x < n / 2; x++)
        a[y * (n / 2) + x] =
            ((a[2 * y * n + 2 * x] + a[2 * y * n + 2 * x + 1]) +
             a[(2 * y + 1) * n + 2 * x] + a[(2 * y + 1) * n + 2 * x + 1]) *
            .25f;
  L[i] = halfq(a[0]);
}
__global__ void depthclip(const float *C, const float *D, const float *M,
                          const float *dd, const float *dm, const float *rec,
                          const float *oldmv, const float *react,
                          const float *comp, float *prep, float *masks, int w,
                          int h, int ow, int oh, Settings s, float kfov,
                          float power) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w;
  float vx = dm[2 * i], vy = dm[2 * i + 1], dist = length(vx * ow, vy * oh);
  float px = ((x + .5f) / w + vx * (dist > .01f)) * w - .5f,
        py = ((y + .5f) / h + vy * (dist > .01f)) * h - .5f;
  int bx = floorf(px), by = floorf(py);
  float fx = px - bx, fy = py - by, zz = z(dd[i], s.camera_near, s.camera_far),
        total = 0, ws = 0;
  for (int k = 0; k < 4; k++) {
    int a = bx + k % 2, b = by + k / 2;
    float wt = (k % 2 ? fx : 1 - fx) * (k / 2 ? fy : 1 - fy),
          pz = z(ld(rec, a, b, w, h), s.camera_near, s.camera_far),
          diff = zz - pz;
    if (valid(a, b, w, h) && wt > .01f && diff > 0) {
      float sep = 1.37e-5f * kfov * length(w, h) * fmaxf(zz, pz);
      total += powf(sat(sep / diff), power) * wt;
      ws += wt;
    }
  }
  float clip = ws > 0 ? sat(1 - total / ws) : 0;
  float d0 = z(ld(rec, x, y - 1, w, h), s.camera_near, s.camera_far),
        d1 = z(rec[i], s.camera_near, s.camera_far),
        d2 = z(ld(rec, x, y + 1, w, h), s.camera_near, s.camera_far);
  if (d0 - d1 > d1 * .01f && d1 - d2 > d2 * .01f)
    clip = 0;
  float nx = M[2 * i], ny = M[2 * i + 1], maxv = length(nx, ny), conv = 1;
  for (int b = -1; b <= 1; b++)
    for (int a = -1; a <= 1; a++) {
      float ox = ld(M, x + a, y + b, w, h, 0, 2, true),
            oy = ld(M, x + a, y + b, w, h, 1, 2, true);
      maxv = fmaxf(maxv, length(ox, oy));
      float v = maxv;
      conv = fminf(conv, (ox / v * nx / v) + (oy / v * ny / v));
    }
  if (length(nx * w, ny * h) <= .01f)
    conv = 1;
  float divergence = sat(1 - conv) * sat(maxv / .01f);
  float pvx = sample(oldmv, (x + .5f) / w + vx, (y + .5f) / h + vy, w, h, 0, 2),
        pvy = sample(oldmv, (x + .5f) / w + vx, (y + .5f) / h + vy, w, h, 1, 2);
  float temporal = dist > 1 ? (1 - sat(length(pvx, pvy) / length(vx, vy))) *
                                  sat(powf(dist / 20, 3))
                            : 0;
  float farz = z(1, s.camera_near, s.camera_far), mind = farz, maxd = 0;
  bool found = false;
  for (int b = -1; b <= 1; b++)
    for (int a = -1; a <= 1; a++) {
      float d = z(ld(dd, x + a, y + b, w, h), s.camera_near, s.camera_far) *
                valid(x + a, y + b, w, h);
      found |= d == farz;
      mind = fminf(mind, d);
      maxd = fmaxf(maxd, d);
    }
  divergence = fmaxf(divergence, sat(temporal - (1 - mind / maxd) * !found));
  float r = 0, t = divergence;
  for (int b = -1; b <= 1; b++)
    for (int a = -1; a <= 1; a++) {
      float dot = 0, n0 = 0, n1 = 0;
      for (int k = 0; k < 3; k++) {
        float c = C[3 * i + k], o = ld(C, x + a, y + b, w, h, k, 3, true);
        dot += c * o;
        n0 += c * c;
        n1 += o * o;
      }
      float p = 1 + (6 - dot / fmaxf(n0, n1) * 6);
      r = fmaxf(r, powf(ld(react, x + a, y + b, w, h, 0, 1, true), p));
      t = fmaxf(t, powf(ld(comp, x + a, y + b, w, h, 0, 1, true), p));
    }
  masks[2 * i] = uq(r);
  masks[2 * i + 1] = uq(t);
  float c[3];
  for (int k = 0; k < 3; k++)
    c[k] = fminf(fmaxf(C[3 * i + k], 0), 65504);
  yc(c);
  for (int k = 0; k < 3; k++)
    prep[4 * i + k] = halfq(c[k]);
  prep[4 * i + 3] = halfq(clip);
}
__global__ void locks(const float *l, float *out, int w, int h, int ow, int oh,
                      float jx, float jy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w, bits = 16;
  float c = l[i], low = 3.402823466e38f, high = 0;
  for (int b = -1; b <= 1; b++)
    for (int a = -1; a <= 1; a++) {
      if (!a && !b)
        continue;
      float v = ld(l, x + a, y + b, w, h, 0, 1, true),
            r = fmaxf(v, c) / fminf(v, c);
      bool same = r > 0 && r < 1.05f;
      if (same)
        bits |= 1 << (3 * (b + 1) + a + 1);
      else {
        low = fminf(low, v);
        high = fmaxf(high, v);
      }
    }
  bool ridge = c > high || c < low;
  int masks[4] = {27, 54, 216, 432};
  for (int k = 0; k < 4; k++)
    ridge &= (bits & masks[k]) != masks[k];
  int a = floorf((x + .5f - jx) / w * ow), b = floorf((y + .5f - jy) / h * oh);
  if (ridge && valid(a, b, ow, oh))
    atomicExch(out + b * ow + a, 1.f);
}
__device__ float lanc(float x) {
  x = fminf(fabsf(x), 2);
  float p = 3.14159265358979323846f * x;
  return x < .001f ? 1 : (sinf(p) / p) * (sinf(p * .5f) / (p * .5f));
}
__device__ void history(const float *a, float u, float v, int w, int h,
                        float *out) {
  float px = fminf(fmaxf(u * w - .5f, 0), float(w)),
        py = fminf(fmaxf(v * h - .5f, 0), float(h));
  int bx = floorf(px), by = floorf(py);
  float wx[4], wy[4], sx = 0, sy = 0, lo[4], hi[4];
  for (int j = 0; j < 4; j++) {
    wx[j] = lanc(px - bx - (j - 1));
    wy[j] = lanc(py - by - (j - 1));
    sx += wx[j];
    sy += wy[j];
    out[j] = 0;
    lo[j] = INFINITY;
    hi[j] = -INFINITY;
  }
  for (int j = 0; j < 4; j++) {
    float row[4] = {};
    for (int k = 0; k < 4; k++) {
      int x = bx + k - 1, y = by + j - 1;
      x = k == 0 ? max(x, 0) : k > 1 ? min(x, w - 1) : x;
      y = j == 0 ? max(y, 0) : j > 1 ? min(y, h - 1) : y;
      for (int c = 0; c < 4; c++) {
        float t = ld(a, x, y, w, h, c, 4);
        row[c] += wx[k] * t;
        if ((j == 1 || j == 2) && (k == 1 || k == 2)) {
          lo[c] = fminf(lo[c], t);
          hi[c] = fmaxf(hi[c], t);
        }
      }
    }
    for (int c = 0; c < 4; c++)
      out[c] += wy[j] * (row[c] / sx);
  }
  for (int c = 0; c < 4; c++)
    out[c] = fminf(fmaxf(out[c] / sy, lo[c]), hi[c]);
}
__device__ bool uvok(float u, float v) {
  return u >= 0 && u <= 1 && v >= 0 && v <= 1;
}
__global__ void accumulate(const float *prep, const float *masks,
                           const float *dm, const float *mip4, const float *nl,
                           const float *old, const float *ol, const float *lh,
                           float *next, float *locknext, float *ln, float *out,
                           float *confidence, int w, int h, int ow, int oh,
                           float jx, float jy, int encoding, bool hasold) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ow * oh)
    return;
  int x = i % ow, y = i / ow;
  float u = (x + .5f) / ow, v = (y + .5f) / oh, lu = u + jx / w,
        lv = v + jy / h;
  int ix = int(u * w), iy = int(v * h);
  float vx = ld(dm, ix, iy, w, h, 0, 2), vy = ld(dm, ix, iy, w, h, 1, 2),
        velocity = length(vx * ow, vy * oh), rx = u + vx, ry = v + vy;
  bool existing = uvok(rx, ry), isnew = !hasold || !existing;
  float dc = sat(sample(prep, lu, lv, w, h, 3, 4)),
        react = sample(masks, lu, lv, w, h, 0, 2),
        accmask = sample(masks, lu, lv, w, h, 1, 2);
  float hist[4] = {}, life = 0, lum0 = 0, temporal = 0;
  bool moving = false;
  if (hasold && existing) {
    history(old, rx, ry, ow, oh, hist);
    temporal = sat(fabsf(hist[3]));
    moving = hist[3] < 0;
    for (int k = 0; k < 3; k++)
      hist[k] = fminf(fmaxf(hist[k], 0), 65504);
    yc(hist);
    life = sample(ol, rx, ry, ow, oh, 0, 2);
    lum0 = sample(ol, rx, ry, ow, oh, 1, 2);
  }
  float thisreact = fmaxf(react, temporal),
        shade = powf(expf(sample(mip4, u, v, max(1, w / 32), max(1, h / 32))),
                     1.f / 6),
        lum = lum0 == 0 ? shade : lum0, diff = 1 - ratio(lum, shade);
  bool newlock = nl[i] > 127.f / 255 && existing && !isnew;
  float updated = newlock ? shade : life <= 1 ? mix(lum, shade, .5f) : lum;
  life = newlock ? (life != 0 ? 2 : 1) : (life > 1 && diff > .1f ? 0 : life);
  thisreact = fmaxf(thisreact, sat((diff - .1f) * 10));
  life = life * (1 - thisreact) * sat(1 - accmask) * (dc < .1f);
  float lockcon = sat(sat(sat(life - 1) * 4) * sat(ratio(updated, shade)));
  float sx = (x + .5f) * (float(w) / ow), sy = (y + .5f) * (float(h) / oh);
  int bx = floorf(sx), by = floorf(sy);
  float ox = bx + .5f - jx - sx, oy = by + .5f - jy - sy;
  bool flipx = ox > 0, flipy = oy > 0;
  float kr = fmaxf(thisreact, float(isnew)),
        bmax = fminf(1.99f, float(ow) / w) * (1 - kr),
        bmin = fmaxf(1, (1 + bmax) * .3f),
        bias = mix(bmax, bmin, fmaxf(.25f * dc, kr)),
        curve = mix(-2, -3, sat(velocity / 50));
  float up[3] = {}, mean[3] = {}, moment[3] = {}, lo[3], hi[3], std[3],
        weight = 0, bw = 0;
  for (int j = 0; j < 9; j++) {
    int a = flipx ? 1 - j % 3 : j % 3 - 1, b = flipy ? 1 - j / 3 : j / 3 - 1;
    float dx = ox + a, dy = oy + b, distance = dx * dx + dy * dy,
          d = fminf(distance * bias * bias, 4), aa = .4f * d - 1,
          bb = .25f * d - 1,
          wt = valid(bx + a, by + b, w, h) *
               ((1.5625f * aa * aa - .5625f) * (bb * bb)),
          boxweight = expf(curve * distance);
    weight += wt;
    bw += boxweight;
    for (int k = 0; k < 3; k++) {
      float t = ld(prep, bx + a, by + b, w, h, k, 4);
      up[k] += t * wt;
      mean[k] += t * boxweight;
      moment[k] += t * t * boxweight;
      lo[k] = j ? fminf(lo[k], t) : t;
      hi[k] = j ? fmaxf(hi[k], t) : t;
    }
  }
  bw = fabsf(bw) > .001f ? bw : 1;
  bool good = weight > .001f;
  for (int k = 0; k < 3; k++) {
    mean[k] /= bw;
    std[k] = sqrtf(fabsf(moment[k] / bw - mean[k] * mean[k]));
    if (good)
      up[k] = fminf(fmaxf(up[k] / weight, lo[k]), hi[k]);
  }
  weight = good ? weight / 12 : 0;
  float current = nearbyintf(mean[0] * 255) * (1.f / 255), L[4] = {};
  bool useluma = fmaxf(fmaxf(dc, accmask), diff) < .1f && !isnew;
  if (useluma)
    for (int k = 0; k < 4; k++)
      L[k] = sample(lh, rx, ry, ow, oh, k, 4);
  float d0 = current - L[0], minimum = fabsf(d0);
  for (int k = 1; k < 4; k++) {
    float d = current - L[k];
    if ((d0 > 0) - (d0 < 0) == (d > 0) - (d < 0))
      minimum = fminf(minimum, fabsf(d));
  }
  float instability = (minimum != fabsf(d0)) * powf(sat(std[0] / .1f), 6);
  instability = (instability > 1.f / 255) * (fabsf(d0) >= 1.f / 255) *
                (1 - fmaxf(accmask, powf(thisreact, 1.f / 6))) * (L[2] != 0);
  float accum = existing * (1 - thisreact) * (1 - dc);
  accum = fminf(
      accum, mix(accum, weight * 10, fmaxf(float(moving), sat(velocity * 10))));
  accum = fminf(accum, mix(accum, weight, sat(velocity / 20)));
  float influence = fminf(20, powf(1 / (float(w) / ow * float(h) / oh), 3)),
        boxscale =
            mix(influence, 1, fmaxf(dc, fmaxf(accmask, sat(velocity / 20))));
  bool outside = false;
  for (int k = 0; k < 3; k++) {
    lo[k] = fmaxf(lo[k], mean[k] - std[k] * boxscale);
    hi[k] = fminf(hi[k], mean[k] + std[k] * boxscale);
    outside |= hist[k] < lo[k] || hist[k] > hi[k];
  }
  float contribution = sat(fmaxf(instability, lockcon) * (1 - sqrtf(react)));
  if (outside) {
    for (int k = 0; k < 3; k++)
      hist[k] = mix(fminf(fmaxf(hist[k], lo[k]), hi[k]), hist[k], contribution);
    accum = mix(fminf(accum, .1f), accum, contribution);
  }
  float resolved[3];
  for (int k = 0; k < 3; k++)
    resolved[k] =
        isnew ? up[k]
              : mix(hist[k], up[k], weight / fmaxf(accum + weight, .001f));
  rgb(resolved);
  for (int k = 0; k < 3; k++) {
    next[4 * i + k] = halfq(resolved[k]);
    float t = resolved[k];
    out[3 * i + k] =
        encoding == 0
            ? (t <= .0031308f ? t * 12.92f : 1.055f * powf(t, 1 / 2.4f) - .055f)
            : t;
  }
  life = uvok(u - vx, v - vy)
             ? fmaxf(life - weight /
                                (int(8 * powf(float(ow) / w, 2)) * (.74f / 12)),
                     0)
             : 0;
  temporal = fminf(thisreact, .99f);
  temporal = fmaxf(temporal, mix(temporal, .4f, sat(velocity)));
  temporal = fmaxf(temporal * temporal, fmaxf(dc * .1f, react));
  if (isnew)
    temporal = 1;
  if (sat(velocity * 10) >= 1)
    temporal = -fmaxf(temporal, .001f);
  next[4 * i + 3] = halfq(temporal);
  locknext[2 * i] = halfq(life);
  locknext[2 * i + 1] = halfq(updated);
  ln[4 * i] = uq(current);
  for (int k = 1; k < 4; k++)
    ln[4 * i + k] = uq(L[k - 1]);
  confidence[i] = sat((1 - dc) * (1 - thisreact));
}
} // namespace
namespace {
__device__ float rcp_medium(float x) {
  float b = __int_as_float(0x7ef19fff - __float_as_int(x));
  return b * (-b * x + 2);
}
__global__ void rcas(const float *history, float *out, int w, int h,
                     float sharpness, int encoding) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w;
  float c[5][3], l[5];
  int xx[5] = {0, -1, 0, 1, 0}, yy[5] = {-1, 0, 0, 0, 1};
  for (int j = 0; j < 5; j++) {
    for (int k = 0; k < 3; k++)
      c[j][k] =
          fminf(fmaxf(ld(history, x + xx[j], y + yy[j], w, h, k, 4), 0), 65504);
    l[j] = c[j][2] * .5f + (c[j][0] * .5f + c[j][1]);
  }
  float lmin = l[0], lmax = l[0];
  for (int j = 1; j < 5; j++) {
    lmin = fminf(lmin, l[j]);
    lmax = fmaxf(lmax, l[j]);
  }
  float noise = 1 - .5f * sat(fabsf(.25f * l[0] + .25f * l[1] + .25f * l[3] +
                                    .25f * l[4] - l[2]) *
                              rcp_medium(lmax - lmin));
  float lobe = -INFINITY;
  for (int k = 0; k < 3; k++) {
    float low = fminf(fminf(c[0][k], c[1][k]), fminf(c[3][k], c[4][k])),
          high = fmaxf(fmaxf(c[0][k], c[1][k]), fmaxf(c[3][k], c[4][k]));
    lobe = fmaxf(lobe, fmaxf(-low / (4 * high), (1 - high) / (4 * low - 4)));
  }
  lobe = fmaxf(fminf(lobe, 0), -.1875f) * exp2f(-2 + 2 * sharpness) * noise;
  for (int k = 0; k < 3; k++) {
    float t = (lobe * c[0][k] + lobe * c[1][k] + lobe * c[4][k] +
               lobe * c[3][k] + c[2][k]) *
              rcp_medium(4 * lobe + 1);
    out[3 * i + k] =
        encoding == 0
            ? (t <= .0031308f ? t * 12.92f : 1.055f * powf(t, 1 / 2.4f) - .055f)
            : t;
  }
}
__global__ void validate_input(const float *c, const float *d, const float *m,
                               const float *r, const float *t, int count,
                               int *error) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  bool good = isfinite(d[i]);
  for (int k = 0; k < 3; k++)
    good &= isfinite(c[3 * i + k]);
  for (int k = 0; k < 2; k++)
    good &= isfinite(m[2 * i + k]);
  if (r)
    good &= isfinite(r[i]) && r[i] >= 0 && r[i] <= 1;
  if (t)
    good &= isfinite(t[i]) && t[i] >= 0 && t[i] <= 1;
  if (!good)
    atomicOr(error, 1);
}
__global__ void validate_output(const float *out, const float *next,
                                const float *lock, const float *lh, int count,
                                int *error) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  bool good = true;
  for (int k = 0; k < 3; k++)
    good &= isfinite(out[3 * i + k]);
  for (int k = 0; k < 4; k++)
    good &= isfinite(next[4 * i + k]) && isfinite(lh[4 * i + k]);
  for (int k = 0; k < 2; k++)
    good &= isfinite(lock[2 * i + k]);
  if (!good)
    atomicOr(error, 2);
}
__global__ void normalized_kernel(const float *d, float *n, float *invalid,
                                  int count, float near, float far) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  float q = far / (near - far),
        device = sat(-q + q * near / fminf(fmaxf(d[i], near), far));
  n[i] = z(device, near, far) / z(1, near, far);
  if (invalid)
    invalid[i] = d[i] < near || d[i] > far || !isfinite(d[i]);
}
__global__ void pixel_motion(const float *dm, float *out, int w, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < w * h) {
    out[2 * i] = dm[2 * i] * w;
    out[2 * i + 1] = dm[2 * i + 1] * h;
  }
}
} // namespace
void normalize_depth(const float *d, float *n, float *invalid, unsigned w,
                     unsigned h, float near, float far, CUstream stream) {
  if (!d || !n || !w || !h || !(near > 0 && far > near) || !std::isfinite(far))
    throw std::invalid_argument("invalid depth normalization");
  normalized_kernel<<<(size_t(w) * h + 127) / 128, 128, 0,
                      reinterpret_cast<cudaStream_t>(stream)>>>(
      d, n, invalid, w * h, near, far);
  ck(cudaGetLastError());
}
struct Engine::Impl {
  int w, h, ow, oh;
  Settings s;
  std::vector<float *> allocations;
  float *C, *D, *M, *dd, *dm, *rec, *lum, *prep, *masks, *mip4, *nl, *hist[2],
      *lock[2], *lh[2], *previous, *pixels;
  int *error;
  bool old = false;
  int index = 0;
  float *alloc(size_t n) {
    float *a;
    ck(cudaMalloc(&a, n * sizeof(float)));
    allocations.push_back(a);
    return a;
  }
  ~Impl() {
    for (auto a : allocations)
      cudaFree(a);
  }
};
Engine::Engine(unsigned w, unsigned h, const Settings &s) : p(new Impl) {
  if (w < 32 || h < 32 || s.output_width < w || s.output_height < h ||
      !(s.camera_near > 0 && s.camera_far > s.camera_near) ||
      !(s.camera_fov_y > 0 && s.camera_fov_y < 3.14159265f) || s.encoding > 1 ||
      !(s.sharpness >= 0 && s.sharpness <= 1) || !std::isfinite(s.camera_far))
    throw std::invalid_argument("FSR2 dimensions/camera/encoding unsupported");
  p->error = reinterpret_cast<int *>(p->alloc(1));
  p->w = w;
  p->h = h;
  p->ow = s.output_width;
  p->oh = s.output_height;
  p->s = s;
  size_t n = size_t(w) * h, o = size_t(p->ow) * p->oh;
  p->C = p->alloc(n * 3);
  p->D = p->alloc(n);
  p->M = p->alloc(n * 2);
  p->dd = p->alloc(n);
  p->dm = p->alloc(n * 2);
  p->rec = p->alloc(n);
  p->lum = p->alloc(n);
  p->prep = p->alloc(n * 4);
  p->masks = p->alloc(n * 2);
  p->mip4 = p->alloc((w / 32) * (h / 32));
  p->nl = p->alloc(o);
  p->previous = p->alloc(n * 2);
  p->pixels = p->alloc(n * 2);
  for (int i = 0; i < 2; i++) {
    p->hist[i] = p->alloc(o * 4);
    p->lock[i] = p->alloc(o * 2);
    p->lh[i] = p->alloc(o * 4);
  }
}
Engine::~Engine() = default;
void Engine::reset() { p->old = false; }
const float *Engine::dilated_motion() const {
  return p->old ? p->pixels : nullptr;
}
#ifdef D5_FSR_TEST_API
const float* Engine::test_resource(unsigned slot) const{const float* a[]={p->hist[p->index],p->lock[p->index],p->lh[p->index],p->prep,p->mip4,p->nl};return slot<6?a[slot]:nullptr;}
#endif
void Engine::process(const float *c, const float *d, const float *m,
                     const float *r, const float *t, float *out,
                     float *confidence, CUstream stream, float jx, float jy,
                     float dt) {
  if (!c || !d || !m || !out || !confidence || !std::isfinite(jx) ||
      !std::isfinite(jy) || !std::isfinite(dt))
    throw std::invalid_argument(
        "FSR2 null input or nonfinite frame parameters");
  auto &a = *p;
  cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  int n = a.w * a.h, o = a.ow * a.oh, b = (n + 127) / 128, bo = (o + 127) / 128,
      next = 1 - a.index;
  ck(cudaMemsetAsync(a.error, 0, sizeof(int), st));
  validate_input<<<b, 128, 0, st>>>(c, d, m, r, t, n, a.error);
  int error = 0;
  ck(cudaMemcpyAsync(&error, a.error, sizeof(int), cudaMemcpyDeviceToHost, st));
  ck(cudaStreamSynchronize(st));
  if (error)
    throw std::invalid_argument(
        "FSR2 nonfinite input or out-of-range mask; state unchanged");
  input<<<b, 128, 0, st>>>(c, d, m, a.C, a.D, a.M, a.rec, a.w, a.h, a.s,
                           !a.old);
  reconstruct<<<b, 128, 0, st>>>(a.C, a.D, a.M, a.dd, a.dm, a.rec, a.lum, a.w,
                                 a.h, a.ow, a.oh);
  int mn = (a.w / 32) * (a.h / 32);
  mip<<<(mn + 31) / 32, 32, 0, st>>>(a.C, a.mip4, a.w, a.h, a.w / 32, a.h / 32,
                                     jx, jy);
  float ty = std::tan(a.s.camera_fov_y * .5f), tx = ty * a.w / a.h,
        cx = 2.f * (a.w / 2) / a.w - 1, cy = 1 - 2.f * (a.h / 2) / a.h,
        kfov = std::sqrt(tx * tx + ty * ty + 1) /
               std::sqrt(cx * cx * tx * tx + cy * cy * ty * ty + 1);
  depthclip<<<b, 128, 0, st>>>(
      a.C, a.D, a.M, a.dd, a.dm, a.rec, a.old ? a.previous : nullptr, r, t,
      a.prep, a.masks, a.w, a.h, a.ow, a.oh, a.s, kfov,
      1 + 2 * std::min(1.f, std::hypot(float(a.w), float(a.h)) /
                                std::hypot(1920.f, 1080.f)));
  ck(cudaMemsetAsync(a.nl, 0, o * sizeof(float), st));
  locks<<<b, 128, 0, st>>>(a.lum, a.nl, a.w, a.h, a.ow, a.oh, jx, jy);
  accumulate<<<bo, 128, 0, st>>>(
      a.prep, a.masks, a.dm, a.mip4, a.nl, a.hist[a.index], a.lock[a.index],
      a.lh[a.index], a.hist[next], a.lock[next], a.lh[next], out, confidence,
      a.w, a.h, a.ow, a.oh, jx, jy, a.s.encoding, a.old);
  if (a.s.sharpness > 0)
    rcas<<<bo, 128, 0, st>>>(a.hist[next], out, a.ow, a.oh, a.s.sharpness,
                             a.s.encoding);
  validate_output<<<bo, 128, 0, st>>>(out, a.hist[next], a.lock[next],
                                      a.lh[next], o, a.error);
  ck(cudaGetLastError());
  ck(cudaMemcpyAsync(&error, a.error, sizeof(int), cudaMemcpyDeviceToHost, st));
  ck(cudaStreamSynchronize(st));
  if (error)
    throw std::runtime_error("FSR2 nonfinite output; state unchanged");
  pixel_motion<<<b, 128, 0, st>>>(a.dm, a.pixels, a.w, a.h);
  ck(cudaMemcpyAsync(a.previous, a.dm, size_t(n) * 2 * sizeof(float),
                     cudaMemcpyDeviceToDevice, st));
  ck(cudaStreamSynchronize(st));
  a.index = next;
  a.old = true;
}
} // namespace dlss5::reconstruction
