#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dlss5::nr::detail
{
using Vec = std::vector<int32_t>;
inline int align(int v, int n)
{
    return (v + n - 1) / n * n;
}
inline constexpr int P[32] = {0,  1,  4,  5,  8,  9,  12, 13, 2,  3,  6,  7,  10, 11, 14, 15,
                              16, 17, 20, 21, 24, 25, 28, 29, 18, 19, 22, 23, 26, 27, 30, 31};
inline int perm(int n)
{
    return P[n % 32] + n / 32 * 32;
}
inline int dep(int v, const std::vector<int> &bits)
{
    uint32_t r = 0;
    for (size_t i = 0; i < bits.size(); ++i)
        r |= ((uint32_t(v) >> i) & 1u) << bits[i];
    return int(r);
}
inline int bits_needed(int v)
{
    int n = 0;
    for (; v; v >>= 1)
        ++n;
    return n;
}
inline int planar(int pixel, int c, int count)
{
    return pixel * 16 + (c / 16) * count * 16 + (c & 1) + ((c & 6) << 1) + ((c & 8) >> 2);
}
inline int raw2(int y, int x, int c, int h, int w)
{
    int t = (y % 8) * 8 + x % 8;
    return ((c / 16) * (h / 8) * (w / 4) + (y / 8) * (w / 4) + (x / 8) * 2 + t / 32) * 512 +
           (t % 32) * 16 + c % 16;
}
inline int blocked(int H, int y, int x, int n, int w)
{
    int nc = (n & 1) | ((n & 6) << 3) | ((n & 8) >> 2) | ((n & 16) >> 1) | ((n & 224) << 4);
    int st = ((y >> 1) & 1) | ((x & 3) << 1) | ((y & 1) << 3);
    if (H == 4)
        return nc + ((st & 1) << 2) + ((st & 14) << 5) + (x / 4 + (w / 4) * (y / 4)) * 2048;
    st |= (x & 4) << 2;
    return nc + ((st & 1) << 2) + ((st & 14) << 5) + ((st & 16) << 8) +
           (x / 8 + align(w, 8) / 8 * (y / 4)) * 8192;
}
inline int canonical4(int r, int c, int n, int h, int w)
{
    int ty = r / 4, tx = c / 4, ly = r % 4, lx = c % 4;
    int y = ty / 2 + (h / 8) * ((lx >> 1) + 2 * (ly >> 1));
    int st = (y & 1) + ((ly & 1) << 1) + ((tx & 1) << 2) + ((lx & 1) << 3) +
             16 * ((tx / 2) + (w / 8) * ((ty & 1) + 2 * (y / 2)));
    static const std::vector<int> cb = {0, 4, 5, 1, 3, 9, 10};
    static const std::vector<int> tb = []() {
        std::vector<int> b;
        for (int i = 0; i < 24; ++i)
            if (std::find(cb.begin(), cb.end(), i) == cb.end())
                b.push_back(i);
        return b;
    }();
    return dep(n, cb) + dep(st, tb);
}
struct Plan
{
    int block, seq, heads, h, w, sx, sy, gx, gy, ph, pw;
};
struct Shape
{
    int h, w, dh, dw;
    Shape(int h_, int w_) : h(h_), w(w_)
    {
        if (h < 64 || w < 64 || h % 64 || w % 64)
            throw std::invalid_argument("NR expects positive 64-aligned H/W >=64");
        if (int64_t(h) * w > INT32_MAX / 128)
            throw std::invalid_argument("NR dimensions exceed signed descriptor addressing");
        dh = align(w / 16, 8) / 2;
        dw = align(h / 16, 8) / 2;
    }
    Plan outer(int b) const
    {
        int H, d;
        if (b <= 4 || b >= 66)
        {
            H = 1;
            d = 2;
        }
        else if (b <= 8 || b >= 62)
        {
            H = 2;
            d = 4;
        }
        else if (b <= 14 || b >= 56)
        {
            H = 4;
            d = 8;
        }
        else
        {
            H = 8;
            d = 16;
        }
        int base = b <= 4    ? 1
                   : b <= 8  ? 5
                   : b <= 14 ? 9
                   : b <= 22 ? 15
                   : b <= 55 ? 48
                   : b <= 61 ? 54
                   : b <= 65 ? 62
                             : 66;
        int phase = (b - base) % 4;
        int sx = phase == 1 || phase == 2 ? -4 : 0, sy = phase == 1 || phase == 3 ? -4 : 0;
        int hh = h / d, ww = w / d;
        return {b,
                b + (b <= 22 ? 2 : 85),
                H,
                hh,
                ww,
                sx,
                sy,
                (ww - sx + 7) / 8,
                (hh - sy + 7) / 8,
                H == 8 ? align(hh, 8) / 2 : hh / 2,
                H == 8 ? align(ww, 8) / 2 : ww / 2};
    }
};
inline Vec wide_map(const Plan &p)
{
    int K = p.heads * 32;
    Vec out(size_t(p.gx) * p.gy * 64 * K, -1);
    for (int by = 0; by < p.gy; ++by)
        for (int bx = 0; bx < p.gx; ++bx)
            for (int t = 0; t < 64; ++t)
            {
                int y = by * 8 + (p.heads == 4 ? ((t >> 4) & 3) * 2 + (t & 1) : t / 8) + p.sy;
                int x = bx * 8 + (p.heads == 4 ? ((t >> 1) & 7) : t & 7) + p.sx;
                if (y < 0 || y >= p.h || x < 0 || x >= p.w)
                    continue;
                for (int n = 0; n < K; ++n)
                    out[((by * p.gx + bx) * 64 + t) * K + n] = blocked(p.heads, y, x, n, p.w);
            }
    return out;
}
struct Pool
{
    Vec input, output;
    int tiles, K;
};
inline Pool pool_metadata(const Plan &p)
{
    int H = p.heads, K = H * 32, N = K * 2, tiles = p.gx * p.gy;
    Vec om;
    if (H == 2)
    {
        om.resize(size_t(tiles) * 64 * K, -1);
        for (int by = 0; by < p.gy; ++by)
            for (int bx = 0; bx < p.gx; ++bx)
                for (int t = 0; t < 64; ++t)
                    for (int c = 0; c < K; ++c)
                    {
                        int n = perm(c);
                        int lab = (n & 3) | ((n & 12) << 2) | ((n & 16) >> 1) | ((n & 32) << 4) |
                                  ((t & 7) << 6) | ((t & 8) >> 1) | ((t & 48) << 6);
                        int qy = (8 * by + p.sy) / 4 + lab / 2048,
                            qx = (8 * bx + p.sx) / 4 + (lab % 2048) / 1024;
                        if (qy >= 0 && qy < p.h / 4 && qx >= 0 && qx < p.w / 4)
                            om[((by * p.gx + bx) * 64 + t) * K + c] =
                                (qy * (p.w / 4) + qx) * 1024 + ((lab % 2048) / 512 % 2) * 512 +
                                lab % 512;
                    }
    }
    else
        om = wide_map(p);
    int extent = std::max(p.h * p.w * K, *std::max_element(om.begin(), om.end()) + 1);
    Vec inverse(extent, -1);
    for (size_t i = 0; i < om.size(); ++i)
        if (om[i] >= 0)
            inverse.at(om[i]) = int(i);
    Vec inv2(4096);
    if (H == 2)
        for (int t = 0; t < 64; ++t)
            for (int c = 0; c < 64; ++c)
            {
                int n = (c & 1) | ((t & 1) << 1) | ((t & 2) << 1) | ((c & 2) << 2) |
                        ((c & 8) << 1) | (t & 32);
                int m = ((t & 4) >> 2) | ((t & 16) >> 3) | (c & 4) | ((c & 16) >> 1) |
                        ((t & 8) << 1) | (c & 32);
                inv2[m * 64 + perm(n)] = t * 64 + c;
            }
    Pool ret{Vec(size_t(tiles) * 16 * 4 * K, -1), Vec(size_t(tiles) * 16 * N, -1), tiles, K};
    Vec counts(tiles, 0), pi(4 * K);
    for (int y = 0; y < p.ph; ++y)
        for (int x = 0; x < p.pw; ++x)
        {
            int maximum = -1;
            for (int leaf = 0; leaf < 4; ++leaf)
                for (int k = 0; k < K; ++k)
                {
                    int id = -1;
                    if (H == 2)
                    {
                        int tc = inv2[(((y % 4) * 4 + x % 4) * 4 + leaf) * 64 + k];
                        id = raw2(y / 4 * 8 + tc / 64 / 8, x / 4 * 8 + tc / 64 % 8, tc % 64, p.h,
                                  p.w);
                    }
                    else if (H == 4)
                    {
                        id = canonical4(y * 2 + (leaf % 2), x * 2 + leaf / 2, k, p.h, p.w);
                    }
                    else
                    {
                        int yy = y * 2 + leaf / 2, xx = x * 2 + leaf % 2;
                        if (yy < p.h && xx < p.w)
                            id = blocked(8, yy, xx, k, p.w);
                    }
                    pi[leaf * K + k] = id;
                    maximum = std::max(maximum, id);
                }
            if (maximum < 0)
                continue;
            int tile = inverse.at(maximum) / (64 * K);
            if (tile < 0 || tile >= tiles)
                throw std::runtime_error("pool owner outside CTA grid");
            int slot = counts[tile]++;
            if (slot >= 16)
                throw std::runtime_error("pool CTA capacity exceeded");
            for (int j = 0; j < 4 * K; ++j)
                ret.input[((tile * 16 + slot) * 4 * K) + j] =
                    pi[j] < 0 ? -1 : inverse.at(pi[j]) % (64 * K);
            int pixel;
            if (H == 2)
            {
                int ty = y / 4, tx = x / 4, ly = y % 4, lx = x % 4;
                int band = ty / 2 + (p.ph / 8) * ((lx >> 1) + 2 * (ly >> 1));
                pixel =
                    tx * 2 + (ty & 1) * (p.pw / 2) + (lx & 1) * p.pw + (ly & 1) + band * (2 * p.pw);
            }
            else if (H == 4)
                pixel = ((y % 2) * 2 + x % 2) * (p.ph * p.pw / 4) + (y / 2) * (p.pw / 2) + x / 2;
            else
                pixel = y * p.pw + x;
            for (int n = 0; n < N; ++n)
                ret.output[(tile * 16 + slot) * N + n] = planar(pixel, n, p.ph * p.pw);
        }
    return ret;
}
struct Up
{
    Vec input, inverse, im;
    int rows, N, tiles;
};
inline Up up_metadata(const Plan &p)
{
    int h = p.h, w = p.w, H = p.heads, N = H * 32;
    Up a{{}, Vec(size_t(h) * (H == 8 ? align(w, 8) : w) * N * 2, 0), {}, 0, N, p.gx * p.gy};
    if (H == 2)
    {
        int ih = h / 2, iw = w / 2;
        Vec rawinv(ih * iw * 128);
        for (int r = 0; r < ih; ++r)
            for (int c = 0; c < iw; ++c)
                for (int n = 0; n < 128; ++n)
                    rawinv.at(blocked(4, r, c, n, iw)) = planar(r * iw + c, n, ih * iw);
        int ip[128];
        for (int n = 0; n < 128; ++n)
            ip[perm(n)] = n;
        a.rows = ih * iw;
        a.input.resize(size_t(a.rows) * 128);
        for (int r = 0; r < ih; ++r)
            for (int c = 0; c < iw; ++c)
            {
                int ty = r / 4, tx = c / 4, ly = r % 4, lx = c % 4;
                int y = ty / 2 + (ih / 8) * ((lx >> 1) + 2 * (ly >> 1));
                int st = (y & 1) | ((ly & 1) << 1) | ((tx & 1) << 2) | ((lx & 1) << 3);
                int token = st + 16 * ((tx / 2) + (iw / 8) * ((ty & 1) + 2 * (y >> 1)));
                for (int n = 0; n < 128; ++n)
                    a.input.at(token * 128 + n) = rawinv.at(canonical4(r, c, ip[n], ih, iw));
            }
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                for (int n = 0; n < 64; ++n)
                {
                    int y = r / 16 + (h / 16) * (n >> 4);
                    int low =
                        (y & 1) | ((r & 1) << 1) | (((c >> 3) & 1) << 2) | (((n >> 2) & 1) << 3);
                    int st = low + 16 * (c / 16 + (w / 16) * (((r >> 3) & 1) + 2 * (y >> 1)));
                    int sn = (n & 1) | (((c >> 1) & 1) << 1) | (((n >> 1) & 1) << 2) |
                             ((c & 1) << 3) | ((n >> 3 & 1) << 4) | ((r >> 2 & 1) << 5);
                    int gate = (n & 1) | ((c & 1) << 1) | ((c >> 1 & 1) << 2) |
                               ((n >> 1 & 1) << 3) | ((n >> 3 & 1) << 4) | ((r >> 2 & 1) << 5);
                    int dest = raw2(r, c, n, h, w);
                    a.inverse.at(dest * 2) = st * 64 + perm(sn);
                    a.inverse.at(dest * 2 + 1) = gate;
                }
    }
    else
    {
        Vec pixels(h * w);
        std::set<int> unique;
        int count = H == 4 ? h * w / 4 : align(h, 8) / 2 * (align(w, 8) / 2);
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
            {
                int pixel;
                if (H == 4)
                {
                    int quarter = ((r / 4) * (w / 4) + c / 4) * 4 + (r >> 1 & 1) * 2 + (c >> 1 & 1);
                    pixel = (quarter % 4) * (count / 4) + quarter / 4;
                }
                else
                    pixel = (r / 2) * (align(w, 8) / 2) + c / 2;
                pixels[r * w + c] = pixel;
                unique.insert(pixel);
            }
        std::map<int, int> which;
        for (int pixel : unique)
        {
            which[pixel] = int(which.size());
            for (int n = 0; n < N * 2; ++n)
                a.input.push_back(planar(pixel, n, count));
        }
        a.rows = int(unique.size());
        for (int r = 0; r < h; ++r)
            for (int c = 0; c < w; ++c)
                for (int n = 0; n < N; ++n)
                {
                    int out = H == 4 ? canonical4(r, c, n, h, w) : blocked(8, r, c, n, w);
                    a.inverse.at(out * 2) = which.at(pixels[r * w + c]) * N + n;
                    a.inverse.at(out * 2 + 1) = n;
                }
        a.im = wide_map(p);
    }
    return a;
}
struct RuntimeTable
{
    std::string name;
    std::vector<unsigned char> bytes;
};
inline std::string runtime_array(std::vector<RuntimeTable> &tables, const std::string &name,
                                 const Vec &v, const std::vector<int> &dims,
                                 const std::string &kind = "int")
{
    RuntimeTable table{name, {}};
    if (kind == "signed char") {
        table.bytes.reserve(v.size());
        for (int n : v) {
            if (n < -128 || n > 127) throw std::runtime_error("NR route exceeds int8");
            table.bytes.push_back(static_cast<unsigned char>(n));
        }
    } else {
        const auto *p = reinterpret_cast<const unsigned char *>(v.data());
        table.bytes.assign(p, p + v.size() * sizeof(int32_t));
    }
    tables.push_back(std::move(table));
    std::ostringstream s;
    s << "extern \"C\" { __device__ __constant__ const " << kind;
    if (dims.size() == 1) s << " *" << name;
    else {
        s << " (*" << name << ")";
        for (size_t i = 1; i < dims.size(); ++i) s << "[" << dims[i] << "]";
    }
    s << "; }\n";
    return s.str();
}
struct Classes
{
    Vec data, ids;
    int count = 0;
};
inline Classes classes(const Vec &v, int width)
{
    std::map<Vec, int> m;
    for (size_t i = 0; i < v.size(); i += width)
        m.emplace(Vec(v.begin() + i, v.begin() + i + width), 0);
    Classes a;
    for (auto &kv : m)
    {
        kv.second = a.count++;
        a.data.insert(a.data.end(), kv.first.begin(), kv.first.end());
    }
    for (size_t i = 0; i < v.size(); i += width)
        a.ids.push_back(m.at(Vec(v.begin() + i, v.begin() + i + width)));
    return a;
}
inline std::string routes(int H, const Pool &ds, const Up &up, std::vector<RuntimeTable> &tables)
{
    int K = H * 32;
    Vec rows(ds.tiles * 64);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = ds.input[i * K] < 0 ? -1 : ds.input[i * K] / K;
    auto d = classes(rows, 64);
    Vec local(up.tiles * 64, -1), global(up.tiles * 16, -1);
    for (int tile = 0; tile < up.tiles; ++tile)
    {
        Vec projected(64, -1);
        std::set<int> owned;
        for (int t = 0; t < 64; ++t)
        {
            int im = up.im[(tile * 64 + t) * K];
            if (im >= 0)
            {
                projected[tile * 0 + t] = up.inverse.at(im * 2) / K;
                owned.insert(projected[t]);
            }
        }
        if (owned.size() > 16)
            throw std::runtime_error("UP CTA projection capacity exceeded");
        int i = 0;
        for (int row : owned)
        {
            global[tile * 16 + i] = up.input.at(row * K * 2) / 16;
            for (int t = 0; t < 64; ++t)
                if (projected[t] == row)
                    local[tile * 64 + t] = i;
            ++i;
        }
    }
    auto u = classes(local, 64);
    std::ostringstream s;
    s << runtime_array(tables, "ds_ids", d.ids, {ds.tiles})
      << runtime_array(tables, "ds_rows", d.data, {d.count, 16, 4}, "signed char")
      << runtime_array(tables, "up_ids", u.ids, {up.tiles})
      << runtime_array(tables, "up_local", u.data, {u.count, 64}, "signed char")
      << runtime_array(tables, "up_global", global, {up.tiles, 16});
    s << "template<int First> __device__ __forceinline__ void collect_pool(Frag (&out)[4][4],u32 "
         "(&a)[4]) {\n int l=threadIdx.x,cid=ds_ids[blockIdx.x];\n";
    for (int first : {0, 2})
    {
        s << " if constexpr(First==" << first << ") {\n";
        for (int word = 0; word < 4; ++word)
        {
            s << " { int row=l/4+" << (word & 1) * 8 << ";\n";
            for (int pair = 0; pair < 2; ++pair)
            {
                s << " {int k=route(" << word / 2 * 16 << "+(l&3)*4+" << pair * 2 << ");\n";
                for (int leaf = 0; leaf < 4; ++leaf)
                {
                    s << " int r" << leaf << "=ds_rows[cid][row][" << leaf << "]; u32 v" << leaf
                      << "=0;\n {int rr=r" << leaf
                      << ",bank=(rr&16)/16*8+(k/8)*2+(rr&8)/8,src=(rr&7)*4+(k&7)/2;\n";
                    // route() moves pair's bit1 into bit3; lane bits stay below
                    // bit3. Thus k/8 == 2*(word/2)+pair for every lane and shape.
                    // Only rr bits4 and3 vary: four architectural banks, not16.
                    const int base_bank = 4 * (word / 2) + 2 * pair;
                    for (int bank : {base_bank, base_bank + 1, base_bank + 8, base_bank + 9})
                        s << " u32 b" << bank << "=__shfl_sync(0xffffffff,out[" << first + bank / 8
                          << "][" << bank % 8 / 2 << "]." << (bank % 2 ? 'y' : 'x')
                          << ",src); if(rr>=0 && rr/32==" << first / 2 << " && bank==" << bank
                          << ")v" << leaf << "=b" << bank << ";\n";
                    s << " }\n";
                }
                s << " if((r0>=0 && r0/32==" << first / 2 << ") || (r1>=0 && r1/32==" << first / 2
                  << ") || (r2>=0 && r2/32==" << first / 2 << ") || (r3>=0 && r3/32==" << first / 2
                  << ")) { half2 "
                     "mean=__hmul2(__hadd2(__hadd2(h2(v0),h2(v1)),__hadd2(h2(v2),h2(v3))),"
                     "constant2(.25f)); a["
                  << word
                  << "] |= "
                     "u32(__nv_cvt_halfraw2_to_fp8x2((__half2_raw)mean,__NV_SATFINITE,__NV_E4M3))<<"
                  << pair * 16 << "; } }\n";
            }
            s << " }\n";
        }
        s << " }\n";
    }
    s << "}\n";
    return s.str();
}
inline std::string addresses(const std::map<int, Pool> &ds, std::vector<RuntimeTable> &tables)
{
    std::string text = "namespace transition_packet {\n";
    for (int H : {2, 4, 8})
    {
        const auto &p = ds.at(H);
        Vec pixels(p.tiles * 16);
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = p.output[i * H * 64] < 0 ? -1 : p.output[i * H * 64] / 16;
        text += runtime_array(tables, "pixels" + std::to_string(H), pixels, {p.tiles, 16});
    }
    text += "__device__ __forceinline__ int pixel(int H,int tile,int row){return "
            "H==2?pixels2[tile][row]:H==4?pixels4[tile][row]:pixels8[tile][row];}\n";
    text += "__device__ __forceinline__ int count(int H){return H==2?(NR_H/8)*(NR_W/8):H==4?(NR_H/16)*(NR_W/16):DH*DW;}\n";
    return text + "__device__ __forceinline__ int planar(int p,int c,int n){return "
                  "p*16+(c/16)*n*16+(c&1)+((c&6)<<1)+((c&8)>>2);}\n}\n";
}
inline Vec raw16(int h, int w, bool decoder = false, bool pool = false, int channels = 512)
{
    std::vector<int> cb = {0, 4, 5, 1, 3, 9, 10, 11, 12, 13};
    cb.resize(bits_needed(channels - 1));
    std::vector<int> tb;
    for (int i = 0; i < 32; ++i)
        if (std::find(cb.begin(), cb.end(), i) == cb.end())
            tb.push_back(i);
    Vec a(size_t(h) * w * channels);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int t = decoder ? y * w + x
                    : pool  ? ((x >> 1) & 1) + ((x & 1) << 1) + (((y >> 1) & 1) << 2) +
                                 ((y & 1) << 3) + 16 * (y / 4 + (h / 4) * (x / 4))
                           : ((y >> 1) & 1) + ((x & 3) << 1) + ((y & 1) << 3) +
                                 16 * (y / 4 + (h / 4) * (x / 4));
            for (int n = 0; n < channels; ++n)
                a[(y * w + x) * channels + n] = dep(n, cb) + dep(t, tb);
        }
    return a;
}
inline Vec pointmap(int h, int w, int channels, bool first = false)
{
    std::vector<int> cb = first ? std::vector<int>{0, 1, 4, 5, 3, 9, 10, 11, 12, 13, 14, 15}
                                : std::vector<int>{0, 4, 5, 1, 3, 9, 10, 11, 12, 13, 14, 15};
    cb.resize(bits_needed(channels - 1));
    Vec a(size_t(h) * w * channels);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            int m = ((x & ~1) | (y & 1)) * h + ((y & ~1) | (x & 1));
            m = (m / 16) * 16 + 2 * (m & 7) + ((m >> 3) & 1);
            for (int n = 0; n < channels; ++n)
                a[(y * w + x) * channels + n] =
                    (m / 16) * (16 * channels) + dep(m % 16, {2, 6, 7, 8}) + dep(n, cb);
        }
    return a;
}
inline Vec halfmap(const Vec &p)
{
    std::vector<int> b = {0, 2, 1, 8, 3, 4, 5, 6, 7};
    for (int i = 9; i < 32; ++i)
        b.push_back(i);
    Vec a;
    a.reserve(p.size());
    for (int v : p)
        a.push_back(dep(v, b));
    return a;
}
inline Vec argsort(const Vec &v)
{
    Vec a(v.size());
    std::iota(a.begin(), a.end(), 0);
    std::sort(a.begin(), a.end(), [&](int x, int y) { return v[x] < v[y]; });
    return a;
}
inline Vec gather(const Vec &a, const Vec &indices)
{
    Vec out;
    out.reserve(indices.size());
    for (int i : indices)
        out.push_back(a.at(i));
    return out;
}
inline Vec repack99(int h, int w)
{
    Vec a(h * w * 1024);
    for (int m = 0; m < h * w; ++m)
    {
        int x = m / h, y = m % h, out = 16 * ((x / 4) * (h / 4) + y / 4) + 4 * (x % 4) + y % 4;
        for (int j = 0; j < 256; ++j)
        {
            auto word = [&](int t) {
                return 4096 * (t / 16) + 16 * (t & 7) + ((t >> 3) & 1) + 128 * (j / 8) +
                       4 * (j & 3) + 2 * ((j >> 2) & 1);
            };
            for (int k = 0; k < 4; ++k)
                a.at(word(out) * 4 + k) = word(m) * 4 + k;
        }
    }
    return a;
}
struct Local
{
    Vec plan, members;
    int gx, gy;
};
inline Local local_packets(int h, int w, int phase)
{
    int dy = (phase == 1 || phase == 2) ? -4 : 0, dx = (phase == 1 || phase == 3) ? -4 : 0;
    int gx = (h - dy + 7) / 8, gy = (w - dx + 7) / 8;
    Local a{Vec(gx * gy * 8 * 4 * 2 * 32, -1), Vec(gx * gy * 64, -1), gx, gy};
    for (int cy = 0; cy < gy; ++cy)
        for (int cx = 0; cx < gx; ++cx)
            for (int t = 0; t < 4; ++t)
            {
                int y4 = 2 * cx + dy / 4 + (t & 1), x4 = 2 * cy + dx / 4 + t / 2;
                if (y4 < 0 || y4 >= h / 4 || x4 < 0 || x4 >= w / 4)
                    continue;
                int cell = (h / 4) * x4 + y4;
                for (int p = 0; p < 8; ++p)
                    for (int k = 0; k < 2; ++k)
                        for (int lane = 0; lane < 32; ++lane)
                            a.plan[((((cy * gx + cx) * 8 + p) * 4 + t) * 2 + k) * 32 + lane] =
                                cell * 8192 + 1024 * p + 512 * k + 16 * lane;
                for (int r = 0; r < 16; ++r)
                {
                    int slot = ((r >> 3) & 1) + 2 * (r & 3) + 8 * ((r >> 2) & 1);
                    a.members[(cy * gx + cx) * 64 + 16 * t + slot] =
                        (4 * y4 + r / 4) * w + 4 * x4 + r % 4;
                }
            }
    return a;
}
struct Packets
{
    Vec sources;
    std::vector<uint16_t> routes;
};
inline Packets packets(const Vec &mapping, int rows, int channels)
{
    Packets p;
    for (int m = 0; m < rows; m += 128)
        for (int k = 0; k < channels; k += 64)
        {
            Vec tile(8192, -1);
            std::set<int> bases;
            for (int r = 0; r < std::min(128, rows - m); ++r)
                for (int c = 0; c < 64; ++c)
                {
                    int v = mapping.at((m + r) * channels + k + c);
                    tile[r * 64 + c] = v;
                    if (v >= 0)
                        bases.insert(v / 16);
                }
            if (bases.size() > 1024)
                throw std::runtime_error("raw packet capacity");
            Vec bv(bases.begin(), bases.end());
            size_t start = p.sources.size();
            p.sources.resize(start + 1024, -1);
            for (size_t j = 0; j < bv.size(); ++j)
                p.sources[start + j] = bv[j] * 16;
            for (int v : tile)
                p.routes.push_back(
                    v < 0 ? 0
                          : uint16_t((std::lower_bound(bv.begin(), bv.end(), v / 16) - bv.begin()) *
                                         16 +
                                     v % 16));
        }
    return p;
}
} // namespace dlss5::nr::detail
