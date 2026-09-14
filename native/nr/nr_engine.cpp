#include "nr_engine.h"
#include "nr_layouts.h"
#include "nr_weight_routes.h"
#include <cstring>
#include <fstream>
#include <limits>
#include <type_traits>
#include <utility>
#include <windows.h>

namespace dlss5::nr::detail
{
std::string source_shallow();
std::string source_heads2();
std::string source_heads4();
std::string source_heads8();
std::string source_utility();
std::string source_deep16();
std::string source_vit();
std::string source_bridge();
using U = uint64_t;
using I = int32_t;
void check(CUresult c)
{
    if (c != CUDA_SUCCESS)
    {
        const char *name = nullptr, *text = nullptr;
        cuGetErrorName(c, &name);
        cuGetErrorString(c, &text);
        throw std::runtime_error(std::string("NR CUDA: ") + (name ? name : "unknown") + " " +
                                 (text ? text : ""));
    }
}
struct ContextScope
{
    CUcontext previous = nullptr;
    bool changed = false;
    explicit ContextScope(CUcontext context)
    {
        check(cuCtxGetCurrent(&previous));
        changed = previous != context;
        if (changed)
            check(cuCtxSetCurrent(context));
    }
    ~ContextScope()
    {
        if (changed)
            cuCtxSetCurrent(previous);
    }
};
struct Storage
{
    CUcontext context;
    std::vector<CUdeviceptr> buffers;
    std::vector<CUmodule> modules;
    explicit Storage(CUcontext c) : context(c)
    {
    }
    ~Storage()
    {
        CUcontext old = nullptr;
        cuCtxGetCurrent(&old);
        cuCtxSetCurrent(context);
        for (auto it = modules.rbegin(); it != modules.rend(); ++it)
            cuModuleUnload(*it);
        for (auto it = buffers.rbegin(); it != buffers.rend(); ++it)
            cuMemFree(*it);
        cuCtxSetCurrent(old);
    }
    CUdeviceptr allocate(size_t bytes, bool zero = false)
    {
        CUdeviceptr p;
        check(cuMemAlloc(&p, bytes));
        try
        {
            buffers.push_back(p);
        }
        catch (...)
        {
            cuMemFree(p);
            throw;
        }
        if (zero)
            check(cuMemsetD8(p, 0, bytes));
        return p;
    }
    CUdeviceptr upload(const void *data, size_t size)
    {
        auto p = allocate(size);
        check(cuMemcpyHtoD(p, data, size));
        return p;
    }
    template <class T> CUdeviceptr upload(const std::vector<T> &v)
    {
        return upload(v.data(), v.size() * sizeof(T));
    }
};
struct BodyWeights
{
    U expand, contract, qkv, projection, bias, ffn_gate, attn_gate, cp, rc, oi, ai, pk;
    uint16_t scale;
};
struct TwoWeights
{
    U expand, reduce, tail, qkv, projection, bias, ffn_gate, attn_gate, scale, p, ip, pk;
};
struct Cross
{
    U pool_input = 0, pool_output = 0, up_inverse = 0, projection = 0, gate = 0, matrix = 0;
    I pool_out = 0, skip = 0;
};
struct Layout
{
    I height, width, grid_x, shift_x, shift_y, flags;
};
struct CompletionRegion
{
    U pointer = 0;
    I count = 0, value = 0;
};
struct Completion
{
    CompletionRegion regions[2]{};
    I count = 0, padding = 0;
};
static_assert(sizeof(BodyWeights) == 104 && sizeof(TwoWeights) == 96 && sizeof(Cross) == 56 &&
              sizeof(Layout) == 24 && sizeof(Completion) == 40);
struct Arg
{
    std::vector<unsigned char> bytes;
    template <class T> Arg(const T &v) : bytes(sizeof(T))
    {
        static_assert(std::is_trivially_copyable_v<T>);
        std::memcpy(bytes.data(), &v, sizeof(T));
    }
    template <class T> void set(T v)
    {
        if (sizeof(T) != bytes.size())
            throw std::logic_error("NR argument ABI");
        std::memcpy(bytes.data(), &v, sizeof(T));
    }
};
using Args = std::vector<Arg>;
struct Dim
{
    unsigned x = 1, y = 1, z = 1;
    Dim(int a = 1, int b = 1, int c = 1) : x(a), y(b), z(c)
    {
    }
};
struct Step
{
    CUfunction fn;
    Dim grid, block;
    unsigned shared;
    Args args;
    void launch(CUstream stream)
    {
        std::array<void *, 32> argv{};
        if (args.size() > argv.size())
            throw std::logic_error("NR argument capacity");
        for (size_t i = 0; i < args.size(); ++i)
            argv[i] = args[i].bytes.data();
        check(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared, stream,
                             argv.data(), nullptr));
    }
};

// NVRTC's stable C ABI. Dynamic loading avoids a link/runtime dependency on the
// system NVRTC (12.2), which cannot compile these unchanged FP8 kernel sources.
struct Compiler
{
    HMODULE builtins = nullptr, dll = nullptr;
    std::filesystem::path root;
    using Program = void *;
    using Create = int(__cdecl *)(Program *, const char *, const char *, int, const char *const *,
                                  const char *const *);
    using Compile = int(__cdecl *)(Program, int, const char *const *);
    using Destroy = int(__cdecl *)(Program *);
    using Size = int(__cdecl *)(Program, size_t *);
    using Get = int(__cdecl *)(Program, char *);
    Create create;
    Compile compile;
    Destroy destroy;
    Size log_size, cubin_size;
    Get log_get, cubin_get;
    template <class T> T bind(const char *n)
    {
        auto p = GetProcAddress(dll, n);
        if (!p)
            throw std::runtime_error(std::string("NVRTC missing ") + n);
        return reinterpret_cast<T>(p);
    }
    static std::filesystem::path find_root()
    {
        wchar_t buf[32768];
        DWORD n = GetEnvironmentVariableW(L"NATIVE_NR_TOOLCHAIN", buf, 32768);
        if (n && n < 32768)
            return std::filesystem::path(buf);
        n = GetEnvironmentVariableW(L"DLSS5_FP8_TOOLCHAIN", buf, 32768);
        if (n && n < 32768)
            return std::filesystem::path(buf);
        return std::filesystem::current_path() / ".toolchains/cuda12.8";
    }
    Compiler()
    {
        root = std::filesystem::absolute(find_root());
        try
        {
            auto bin = root / "nvrtc/bin";
            builtins =
                LoadLibraryExW((bin / "nvrtc-builtins64_128.dll").c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            dll =
                LoadLibraryExW((bin / "nvrtc64_120_0.dll").c_str(), nullptr,
                               LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!builtins || !dll)
                throw std::runtime_error("NR needs NVRTC 12.8: set NATIVE_NR_TOOLCHAIN (nvrtc/bin, "
                                         "runtime/include, cccl/include)");
            auto version = bind<int(__cdecl *)(int *, int *)>("nvrtcVersion");
            int major = 0, minor = 0;
            if (version(&major, &minor) || major != 12 || minor != 8)
                throw std::runtime_error("NR requires NVRTC 12.8");
            create = bind<Create>("nvrtcCreateProgram");
            compile = bind<Compile>("nvrtcCompileProgram");
            destroy = bind<Destroy>("nvrtcDestroyProgram");
            log_size = bind<Size>("nvrtcGetProgramLogSize");
            cubin_size = bind<Size>("nvrtcGetCUBINSize");
            log_get = bind<Get>("nvrtcGetProgramLog");
            cubin_get = bind<Get>("nvrtcGetCUBIN");
        }
        catch (...)
        {
            if (dll)
                FreeLibrary(dll);
            if (builtins)
                FreeLibrary(builtins);
            throw;
        }
    }
    ~Compiler()
    {
        FreeLibrary(dll);
        FreeLibrary(builtins);
    }
    std::vector<char> image(const std::string &name, const std::string &source,
                            const std::map<std::string, int> &defines,
                            const std::map<std::string, std::string> &headers)
    {
        std::vector<const char *> hn, hv;
        for (const auto &h : headers)
        {
            hn.push_back(h.first.c_str());
            hv.push_back(h.second.c_str());
        }
        Program p = nullptr;
        int result = create(&p, source.c_str(), name.c_str(), int(hn.size()), hv.data(), hn.data());
        if (result)
            throw std::runtime_error("nvrtcCreateProgram failed: " + std::to_string(result));
        struct Guard
        {
            Compiler *c;
            Program *p;
            ~Guard()
            {
                c->destroy(p);
            }
        } guard{this, &p};
        std::vector<std::string> options = {
            "--gpu-architecture=sm_89", "--std=c++17", "--device-as-default-execution-space",
            "-I" + (root / "runtime/include").string(), "-I" + (root / "cccl/include").string()};
        for (const auto &v : defines)
            options.push_back("-D" + v.first + "=" + std::to_string(v.second));
        std::vector<const char *> opts;
        for (auto &s : options)
            opts.push_back(s.c_str());
        result = compile(p, int(opts.size()), opts.data());
        if (result)
        {
            size_t n = 0;
            log_size(p, &n);
            std::string log(n, '\0');
            if (n)
                log_get(p, log.data());
            throw std::runtime_error(name + ": NVRTC " + std::to_string(result) + "\n" + log);
        }
        size_t n = 0;
        if (cubin_size(p, &n) || !n)
            throw std::runtime_error("NVRTC empty cubin");
        std::vector<char> data(n);
        if (cubin_get(p, data.data()))
            throw std::runtime_error("NVRTC cubin retrieval failed");
        return data;
    }
};
inline uint16_t to_half(float value)
{
    uint32_t x;
    std::memcpy(&x, &value, 4);
    uint16_t sign = uint16_t(x >> 16) & 0x8000;
    int exp = int((x >> 23) & 255) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp >= 31)
        return sign | 0x7c00 | (mant && ((x >> 23) & 255) == 255 ? 0x200 : 0);
    if (exp <= 0)
    {
        if (exp < -10)
            return sign;
        mant |= 0x800000;
        int shift = 14 - exp;
        uint32_t half = mant >> shift, rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
        return sign | uint16_t(half + (rem > mid || (rem == mid && (half & 1))));
    }
    uint32_t half = (uint32_t(exp) << 10) | (mant >> 13), rem = mant & 8191;
    return sign | uint16_t(half + (rem > 4096 || (rem == 4096 && (half & 1))));
}
struct Weights
{
    Storage mem;
    std::map<std::string, std::vector<unsigned char>> records;
    std::map<std::pair<int, int>, U> pointers;
    std::map<int, BodyWeights> one_cache;
    std::map<int, TwoWeights> two_cache;
    U adapter = 0, main_gate = 0, skip_gate = 0, readout = 0, up_gate = 0;
    template <class T> static T get(const std::vector<unsigned char> &v, size_t i)
    {
        if (i > v.size() || sizeof(T) > v.size() - i)
            throw std::runtime_error("Truncated NR weight record");
        T r;
        std::memcpy(&r, v.data() + i, sizeof(T));
        return r;
    }
    Weights(CUcontext c, const std::filesystem::path &dir) : mem(c)
    {
        std::ifstream f(dir / "weights_ht_blob.bin", std::ios::binary | std::ios::ate);
        if (!f)
            throw std::runtime_error("Cannot read original model/weights_ht_blob.bin");
        auto size = f.tellg();
        if (size < 8 || size > std::streamoff(1ULL << 30))
            throw std::runtime_error("Invalid frozen BIN size");
        std::vector<unsigned char> blob(static_cast<size_t>(size));
        f.seekg(0);
        if (!f.read(reinterpret_cast<char *>(blob.data()), size))
            throw std::runtime_error("Incomplete frozen BIN");
        if (get<uint64_t>(blob, 0) != blob.size())
            throw std::runtime_error("Frozen BIN length mismatch");
        size_t cursor = 8;
        while (cursor < blob.size())
        {
            auto length = get<uint64_t>(blob, cursor);
            cursor += 8;
            if (length < 1 || length > 4096 || length > blob.size() - cursor)
                throw std::runtime_error("Invalid frozen BIN name");
            std::string name(reinterpret_cast<const char *>(blob.data() + cursor), size_t(length));
            cursor += size_t(length);
            auto span = get<uint64_t>(blob, cursor);
            cursor += 8;
            size_t body = cursor;
            if (span < 40 || span > blob.size() - cursor)
                throw std::runtime_error("Invalid frozen BIN body");
            cursor += size_t(span);
            auto total = get<uint64_t>(blob, body), bytes = get<uint64_t>(blob, body + 8);
            size_t start = body + 20;
            if (total != span || bytes % 2 || bytes > span - 36)
                throw std::runtime_error("Invalid frozen BIN payload");
            size_t end = start + size_t(bytes);
            auto rank = get<uint64_t>(blob, end + 8);
            if (rank > 16 || end + 16 + 4 * rank != cursor || records.count(name))
                throw std::runtime_error("Invalid frozen BIN trailer");
            records[name] = std::vector<unsigned char>(blob.begin() + start, blob.begin() + end);
        }
        if (records.size() != 153)
            throw std::runtime_error("Expected 153 frozen BIN records");
        for (int b = 23; b < 48; ++b)
        {
            int count = b == 39 ? 1 : (b >= 30 && b <= 38 ? 5 : 4);
            for (int l = 0; l < count; ++l)
            {
                if (b >= 31 && b <= 38 && l == 3)
                    continue;
                static const size_t sw[] = {0x80000, 0x40400, 0xe0040, 0x40400, 0x80010},
                                    vit[] = {0x400010, 0x400800, 0x300080, 2, 0x100800};
                size_t expected = b <= 30 || b >= 40 ? sw[l] : b == 39 ? 0x80400 : vit[l];
                if (raw(b, l).size() != expected)
                    throw std::runtime_error("Unexpected deep frozen weight size");
                ptr(b, l);
            }
        }
        std::vector<uint16_t> ad(16 * 32);
        int q_to_n[32] = {0,  1,  8,  9,  2,  3,  10, 11, 4,  5,  12, 13, 6,  7,  14, 15,
                          16, 17, 24, 25, 18, 19, 26, 27, 20, 21, 28, 29, 22, 23, 30, 31};
        for (int row = 0; row < 16; ++row)
            for (int n = 0; n < 32; ++n)
            {
                int column = q_to_n[channel_inverse[n]], frag = column / 8;
                ad[row * 32 + n] = hmma(0, frag < 2 ? 0x2010 : 0x2210, frag % 2, row, column % 8);
            }
        adapter = mem.upload(ad);
        std::vector<uint16_t> mg(32), sg(32), ro(32 * 8), gate(32);
        for (int n = 0; n < 32; ++n)
        {
            mg[n] = get<uint16_t>(raw(70), 0x2050 + post_gate_route[n] * 2);
            sg[n] = get<uint16_t>(raw(70), 0x2090 + post_gate_route[n] * 2);
            gate[n] = get<uint16_t>(raw(66), 0x2860 + post_gate_route[n] * 2);
            for (int j = 0; j < 8; ++j)
            {
                ro[n * 8 + j] = hmma(70, n < 16 ? 0x5130 : 0x5330, 0, n % 16, j);
                if (j >= 4 && (ro[n * 8 + j] & 0x7fff))
                    throw std::runtime_error("Nonzero unused post HMMA columns");
            }
        }
        main_gate = mem.upload(mg);
        skip_gate = mem.upload(sg);
        readout = mem.upload(ro);
        up_gate = mem.upload(gate);
    }
    const std::vector<unsigned char> &raw(int b, int l = 0) const
    {
        return records.at("block" + std::to_string(b) + ".layer" + std::to_string(l) + ".layer");
    }
    U ptr(int b, int l = 0)
    {
        auto key = std::make_pair(b, l);
        auto it = pointers.find(key);
        if (it != pointers.end())
            return it->second;
        return pointers[key] = mem.upload(raw(b, l));
    }
    uint16_t hmma(int b, int offset, int pair, int row, int col) const
    {
        int lane = col * 4 + (row % 8) / 2, slot = row % 2 + (row / 8) * 2;
        return get<uint16_t>(raw(b), offset + 2 * (lane * 8 + pair * 4 + slot));
    }
    BodyWeights one(int b)
    {
        auto it = one_cache.find(b);
        if (it != one_cache.end())
            return it->second;
        U p = ptr(b);
        int shift = b == 0    ? 0x400
                    : b == 66 ? 0x840
                    : b == 70 ? 0x70
                              : 0,
            gate = b == 0    ? 0x2410
                   : b == 66 ? 0x2810
                             : 0x2010;
        auto scale = to_half(get<float>(raw(b), 0x4c60 + shift));
        return one_cache[b] = {p,
                               p + 0x1000,
                               p + 0x2060 + shift,
                               p + 0x4c70 + shift,
                               p + 0x2c60 + shift,
                               p + gate,
                               p + 0x5070 + shift,
                               0,
                               0,
                               0,
                               0,
                               0,
                               scale};
    }
    TwoWeights two(int b)
    {
        auto it = two_cache.find(b);
        if (it != two_cache.end())
            return it->second;
        U p = ptr(b);
        int shift = b == 62 ? 0x2060 : 0;
        std::vector<uint16_t> scales = {to_half(get<float>(raw(b), 0xe0a0 + shift)),
                                        to_half(get<float>(raw(b), 0xe0a4 + shift))};
        return two_cache[b] = {p,
                               p + 0x4000,
                               p + 0x6000,
                               p + 0x70a0 + shift,
                               p + 0xe0b0 + shift,
                               p + 0xa0a0 + shift,
                               p + (b == 62 ? 0x9000 : 0x7010),
                               p + 0xf0b0 + shift,
                               mem.upload(scales),
                               0,
                               0,
                               0};
    }
};

struct State
{
    Shape shape;
    Storage mem;
    Compiler &compiler;
    Weights &wt;
    std::map<std::string, CUmodule> mods;
    std::map<std::string, std::string> headers;
    std::vector<Step> enc, deep, dec;
    Step clear{}, pre{}, post{};
    U arena = 0, status = 0, completion_slab = 0;
    size_t completion_count = 0;
    int *host_status = nullptr;
    std::map<std::string, I> offsets;
    std::map<std::string, U> aux;
    struct Patch
    {
        size_t step, arg;
    };
    std::vector<Patch> completion_patches;
    explicit State(Shape s, CUcontext context, Compiler &c, Weights &w)
        : shape(s), mem(context), compiler(c), wt(w)
    {
        try
        {
            build_outer();
            build_deep();
            completion_slab = mem.allocate(completion_count * 4);
            for (const auto &p : completion_patches)
            {
                Completion v;
                std::memcpy(&v, deep[p.step].args[p.arg].bytes.data(), sizeof(v));
                for (int i = 0; i < v.count; ++i)
                    v.regions[i].pointer += completion_slab;
                deep[p.step].args[p.arg].set(v);
            }
            check(cuMemHostAlloc(reinterpret_cast<void **>(&host_status), sizeof(int), 0));
            check(cuCtxSynchronize());
        }
        catch (...)
        {
            if (host_status)
                cuMemFreeHost(host_status);
            throw;
        }
    }
    ~State()
    {
        if (host_status)
            cuMemFreeHost(host_status);
    }
    CUmodule module(const std::string &name)
    {
        auto it = mods.find(name);
        if (it != mods.end())
            return it->second;
        std::string src;
        if (name == "shallow")
            src = source_shallow();
        else if (name == "heads2")
            src = source_heads2();
        else if (name == "heads4")
            src = source_heads4();
        else if (name == "heads8")
            src = source_heads8();
        else if (name == "utility")
            src = source_utility();
        else if (name == "deep16")
            src = source_deep16();
        else if (name == "vit")
            src = source_vit();
        else if (name == "bridge")
            src = source_bridge();
        else
            throw std::logic_error("Unknown NR module");
        bool isdeep = name == "deep16" || name == "vit" || name == "bridge";
        int h = shape.dh, w = shape.dw, ph = align(h, 8) / 2, pw = align(w, 8) / 2, t = ph * pw;
        std::map<std::string, int> defines =
            isdeep ? std::map<std::string, int>{{"DH", h},
                                                {"DW", w},
                                                {"SH", (h + 7) / 8},
                                                {"PW", pw},
                                                {"PH", ph},
                                                {"DT", t},
                                                {"DP", align(t, 128)},
                                                {"MT", (t + 127) / 128}}
                   : std::map<std::string, int>{{"NR_H", shape.h}, {"NR_W", shape.w}};
        auto image = compiler.image(name + ".cu", src, defines,
                                    isdeep ? std::map<std::string, std::string>{} : headers);
        CUmodule mod;
        check(cuModuleLoadData(&mod, image.data()));
        try
        {
            mem.modules.push_back(mod);
        }
        catch (...)
        {
            cuModuleUnload(mod);
            throw;
        }
        mods[name] = mod;
        return mod;
    }
    Step step(const std::string &unit, const std::string &name, Dim grid, Dim block,
              unsigned shared, Args args)
    {
        CUfunction f;
        check(cuModuleGetFunction(&f, module(unit), name.c_str()));
        return {f, grid, block, shared, std::move(args)};
    }
    U buffer(size_t bytes)
    {
        return mem.allocate(bytes, true);
    }
    U map(const Vec &v)
    {
        return mem.upload(v);
    }
    Completion completion(int count, bool split = false, int value = 3, bool first_only = false)
    {
        Completion c;
        c.count = split && !first_only ? 2 : 1;
        for (int i = 0; i < c.count; ++i)
        {
            c.regions[i] = {U(completion_count * 4), I(count), I(split && i == 0 ? value : 0)};
            completion_count += count;
        }
        return c;
    }
    void add(const std::string &unit, const std::string &name, Dim grid, Dim block, Args args,
             bool comp = false)
    {
        if (comp)
            completion_patches.push_back({deep.size(), args.size() - 1});
        deep.push_back(step(unit, name, grid, block, 0, std::move(args)));
    }
    void flat(const std::string &unit, const std::string &name, int count, Args args,
              bool comp = false)
    {
        add(unit, name, {(count + 255) / 256}, {256}, std::move(args), comp);
    }
    void repack(U source, U target, U imap, U omap, int count, int done = 0)
    {
        flat("bridge", "raw_repack_completion", count,
             {source, target, imap, omap, I(count), done ? completion(done) : Completion{}},
             done != 0);
    }
    void build_outer();
    void build_deep();
};
void State::build_outer()
{
    int h = shape.h, w = shape.w;
    std::map<int, Pool> ds;
    std::map<int, Up> up;
    for (auto v : {std::pair<int, int>{2, 8}, {4, 14}, {8, 22}})
        ds.emplace(v.first, pool_metadata(shape.outer(v.second)));
    for (auto v : {std::pair<int, int>{2, 62}, {4, 56}, {8, 48}})
        up.emplace(v.first, up_metadata(shape.outer(v.second)));
    headers["routes4.cuh"] = routes(4, ds.at(4), up.at(4));
    headers["routes8.cuh"] = routes(8, ds.at(8), up.at(8));
    headers["addresses.cuh"] = addresses(shape, ds);
    size_t cursor = 0;
    auto reserve = [&](const std::string &name, size_t bytes) {
        cursor = (cursor + 255) / 256 * 256;
        if (cursor + bytes >= INT32_MAX)
            throw std::invalid_argument("Outer arena exceeds int32 offsets");
        offsets[name] = I(cursor);
        cursor += bytes;
    };
    auto key = [](const char *s, int b) { return std::string(s) + std::to_string(b); };
    reserve("skip0", size_t(h) * w * 32);
    reserve("out0", size_t(h) * w / 4 * 32);
    std::vector<int> blocks;
    for (int b = 1; b <= 22; ++b)
        blocks.push_back(b);
    for (int b = 48; b <= 69; ++b)
        blocks.push_back(b);
    for (int b : blocks)
    {
        auto p = shape.outer(b);
        int padded = p.heads == 8 && b != 55 ? align(p.w, 8) : p.w;
        reserve(key("out", b), size_t(p.h) * padded * p.heads * 32);
        if (b == 4 || b == 8 || b == 14 || b == 22)
            reserve(key("pool", b), size_t(p.ph) * p.pw * p.heads * 64);
    }
    reserve("deep_out", size_t(shape.dh) * shape.dw * 512);
    cursor = (cursor + 255) / 256 * 256;
    size_t counter_begin = cursor;
    for (int b : blocks)
    {
        auto p = shape.outer(b);
        reserve(key("counter", b), size_t(p.gx) * p.gy * 4);
    }
    arena = mem.allocate(cursor);
    status = mem.allocate(4);
    int n = int((cursor - counter_begin) / 4);
    clear = step("utility", "clear_counters", {(n + 255) / 256}, {256}, 0,
                 {U(arena + counter_begin), I(n), status});
    aux["DS_pre"] = mem.allocate(size_t(h) * w / 4 * 32 * 2);
    aux["UP_mixed"] = mem.allocate(size_t(h) * w / 4 * 32 * 2);
    aux["UP147_projection"] = mem.allocate(size_t(h) * w / 64 * 64 * 2);
    aux["UP141_projection"] = mem.allocate(size_t(h) * w / 256 * 128 * 2);
    aux["UP133_projection"] = mem.allocate(size_t(shape.dh) * shape.dw * 256 * 2);
    pre = step("shallow", "block0_native_packet", {w / 8, h / 8}, {32}, 0,
               {wt.one(0), U(0), wt.adapter, U(arena + offsets.at("skip0")),
                U(arena + offsets.at("out0")), status});
    I previous = offsets.at("out0");
    for (int b : blocks)
    {
        auto p = shape.outer(b);
        int H = p.heads;
        if (b == 48)
            previous = offsets.at("deep_out");
        I out = offsets.at(key("out", b)), counter = offsets.at(key("counter", b));
        auto &steps = b <= 22 ? enc : dec;
        Dim grid(p.gx * p.gy), block(32, H);
        Cross cross;
        Args values;
        std::string unit, name;
        unsigned shared = 0;
        if (H == 1)
        {
            values = {arena, wt.one(b), U(0), U(0), previous, out, counter, U(0), U(0)};
            unit = "shallow";
            name = "outer1_static_" + std::to_string(p.seq);
            if (b == 4)
            {
                name = "outer1_static_ds";
                values[8].set(aux.at("DS_pre"));
                Args extra = {U(wt.ptr(b) + 0x50b0), U(0), U(0), U(0), offsets.at("pool4")};
                values.insert(values.end(), extra.begin(), extra.end());
            }
            if (b == 66)
            {
                name = "outer1_static_up";
                values[7].set(aux.at("UP_mixed"));
                Args extra = {U(wt.ptr(66) + 0x2000), U(0), U(0), U(0), U(0), wt.up_gate,
                              offsets.at("out4")};
                values.insert(values.end(), extra.begin(), extra.end());
            }
        }
        else
        {
            if (b == 8 || b == 14 || b == 22)
            {
                cross.pool_input = map(ds.at(H).input);
                cross.pool_output = map(ds.at(H).output);
                cross.pool_out = offsets.at(key("pool", b));
                cross.matrix = wt.ptr(b) + (H == 2 ? 0xf130 : H == 4 ? 0x30230 : 0xa8440);
            }
            U ui_ptr = 0;
            if (b == 48 || b == 56 || b == 62)
            {
                cross.up_inverse = map(up.at(H).inverse);
                cross.projection = aux.at("UP" + std::to_string(p.seq) + "_projection");
                cross.skip = offsets.at(key("out", b == 48 ? 22 : b == 56 ? 14 : 8));
                cross.gate = wt.ptr(b) + (H == 2 ? 0x9080 : H == 4 ? 0x20100 : 0x78200);
                ui_ptr = map(up.at(H).input);
                if (b == 62)
                    steps.push_back(step("heads2", "streamed_project2", {(up.at(H).rows + 15) / 16},
                                         {32, 2}, 0,
                                         {arena, cross.projection, U(wt.ptr(b) + 0x7000), ui_ptr,
                                          previous, I(up.at(H).rows)}));
            }
            if (H == 2)
            {
                U meta = map({p.h, p.w, p.gx, p.sx, p.sy, p.seq});
                unit = "heads2";
                name = b == 8 ? "packet2_ds" : "streamed2";
                values = {arena, wt.two(b), meta, U(0), previous, out, counter, cross};
                shared = b == 8 ? 12288 : 4096;
            }
            else
            {
                int flags = b == 9 || b == 15 ? 1 : b == 55 || b == 61 ? 2 : 0;
                Layout layout{p.h, p.w, p.gx, p.sx, p.sy, flags};
                values = {arena, wt.ptr(b), layout, previous, out, counter, I(b == 48 || b == 56),
                          arena, cross};
                unit = H == 4 ? "heads4" : "heads8";
                if (b == 14 || b == 22)
                {
                    name = "packet" + std::to_string(H) + "_ds";
                    values.push_back(U(0));
                }
                else if (b == 48 || b == 56)
                {
                    name = "packet" + std::to_string(H) + "_up";
                    values.push_back(ui_ptr);
                }
                else if (H == 4)
                    name = flags == 1 ? "four_input" : flags == 2 ? "four_output" : "four_ordinary";
                else
                    name = flags == 1   ? "full8_inpview"
                           : flags == 2 ? "full8_outview"
                                        : "full8_chained";
                shared = H * 2048;
            }
        }
        steps.push_back(step(unit, name, grid, block, shared, std::move(values)));
        previous = b == 4 || b == 8 || b == 14 || b == 22 ? offsets.at(key("pool", b)) : out;
        if (b == 22)
        {
            int count = p.ph * p.pw * 512;
            steps.push_back(step("utility", "pool8_padding", {(count + 255) / 256}, {256}, 0,
                                 {arena, previous, I(p.h), I(p.w), I(p.ph), I(p.pw)}));
        }
    }
    post = step("shallow", "post70_native_head", {w / 8 + 1, h / 8 + 1}, {32}, 0,
                {wt.one(70), U(arena + previous), U(arena + offsets.at("skip0")), wt.main_gate,
                 wt.skip_gate, wt.readout, U(0), status});
}
void State::build_deep()
{
    int h = shape.dh, w = shape.dw, ph = align(h, 8) / 2, pw = align(w, 8) / 2, t = ph * pw,
        rows = h * w, dp = align(t, 128), sh = (h + 7) / 8, sw = (w + 7) / 8, mt = (t + 127) / 128;
    if (std::max(int64_t(rows) * 512, int64_t(dp) * 4096 * 8) > INT32_MAX)
        throw std::invalid_argument("Deep addressing exceeds int32");
    auto encmap = raw16(h, w), decmap = raw16(h, w, true), pc = pointmap(ph, pw, 1024),
         pi = pointmap(ph, pw, 1024, true), ff = pointmap(ph, pw, 4096),
         pool = raw16(ph, pw, false, true), terminal = raw16(ph, pw, false, true, 1024);
    Vec ri(1024), rc(1024);
    for (int n = 0; n < 1024; ++n)
    {
        ri[n] = dep(n, {0, 1, 3, 4, 2, 5, 6, 7, 8, 9});
        rc[n] = dep(n, {0, 3, 4, 1, 2, 5, 6, 7, 8, 9});
    }
    auto nextroute = gather(argsort(rc), ri);
    Vec nextids(t * 1024);
    for (int r = 0; r < t; ++r)
        for (int n = 0; n < 1024; ++n)
            nextids[r * 1024 + n] = r * 1024 + nextroute[n];
    U a = buffer(rows * 512), b = buffer(rows * 512), att = buffer(rows * 512),
      c = buffer(rows * 512), poolraw = buffer(t * 512), skip56 = buffer(rows * 512);
    struct PlanPtr
    {
        U plan, members;
        Dim grid;
    };
    std::array<PlanPtr, 4> plans;
    for (int phase = 0; phase < 4; ++phase)
    {
        auto p = local_packets(h, w, phase);
        plans[phase] = {map(p.plan), map(p.members), {p.gx, p.gy, 4}};
    }
    auto swin = [&](int block, U current) {
        int decoder = block >= 40;
        bool first = block == 23;
        add("deep16", first ? "first25_completion" : "chained_completion", {sh, sw, 2},
            {32, first ? 4 : 8},
            {current, wt.ptr(block, 0), U(0), a, I(decoder), completion(sh * sw * 2)}, true);
        add("deep16", first ? "special26_completion" : "project4_completion", {sh * 2, sw}, {32, 4},
            {a, wt.ptr(block, 1), current, U(0), U(0), b, U(0), U(0), I(decoder),
             completion(sh * sw * 2)},
            true);
        auto p = plans[(block - (decoder ? 40 : 23)) % 4];
        add("deep16", "joint_local64_completion", p.grid, {32, 4},
            {b, p.plan, wt.ptr(block, 2), U(0), att, U(0), p.members,
             completion(p.grid.x * p.grid.y * p.grid.z)},
            true);
        std::string name = block == 30 ? "pool56" : block == 47 ? "out132" : "project8_completion";
        U out = block == 47 ? arena + offsets.at("deep_out") : c;
        Args values = {
            att,       wt.ptr(block, 3), b, U(0), U(0), out, U(0), U(block == 30 ? poolraw : 0),
            I(decoder)};
        bool done = block != 30 && block != 47;
        if (done)
            values.push_back(completion(sh * sw * 2));
        add("deep16", name, {sh * 2, sw}, {32, done ? 8 : 4}, std::move(values), done);
        return out;
    };
    U current = arena + offsets.at("pool22");
    for (int block = 23; block < 31; ++block)
        current = swin(block, current);
    repack(current, skip56, map(encmap), 0, rows * 512, sh * sw * 2);
    U x0 = buffer(dp * 1024), x1 = buffer(dp * 1024), q = buffer(dp * 1024), k = buffer(dp * 1024),
      v = buffer(dp * 1024), at = buffer(dp * 1024), hidden = buffer(dp * 4096),
      partial = buffer(t * 1024 * 4 * 2), scratch = buffer(dp * 1024 * 3 * 2),
      hp = map(halfmap(pc)), pcm = map(pc);
    auto matrix_args = [&](U source, U weight, int R, int K, int N, int mode, int splits, U raw = 0,
                           U outmap = 0, U part = 0, U skip = 0, int gate = 0,
                           const Vec *source_map = nullptr) {
        U pkt = 0, route = 0;
        if (source_map)
        {
            auto p = packets(*source_map, R, K);
            pkt = map(p.sources);
            route = mem.upload(p.routes);
        }
        return Args{source, weight,  skip,      U(0),    U(0), part, I(R),  I(K),
                    I(N),   I(mode), I(splits), I(gate), I(0), U(0), I(0),  raw,
                    outmap, U(0),    U(0),      U(0),    U(0), pkt,  route, U(0)};
    };
    add("bridge", "matrix_phase", {mt, 8}, {128},
        matrix_args(poolraw, wt.ptr(30, 4), t, 512, 1024, 1, 1, x0, map(terminal), 0, 0, 0, &pool));
    repack(x0, x1, map(gather(terminal, nextids)), map(pi), t * 1024);
    current = x1;
    U ffmap = map(ff), nextpm = map(gather(pi, argsort(nextids)));
    for (int block = 31; block < 39; ++block)
    {
        auto args = matrix_args(current, wt.ptr(block, 0), t, 1024, 4096, 0, 1, hidden, ffmap);
        args.push_back(I(64));
        args.push_back(completion(mt * 32));
        add("vit", "matrix_expand_completion", {mt * 32}, {32, 4}, std::move(args), true);
        args = matrix_args(hidden, wt.ptr(block, 1), t, 4096, 1024, 2, 4, 0, 0, partial, current,
                           0x400000);
        args.push_back(I(64));
        add("vit", "matrix_contract", {mt * 8, 1, 4}, {32, 4}, std::move(args));
        flat("bridge", "publish_split", t * 1024, {partial, scratch, hp, I(t * 1024), I(4)});
        flat("bridge", "merge_phase_completion", t * 1024,
             {partial, scratch, hp, U(0), I(t * 1024), I(4), U(0), I(0), x0, pcm,
              completion(mt * 8, true)},
             true);
        args = {x0,
                wt.ptr(block, 2),
                U(0),
                U(0),
                U(0),
                U(0),
                scratch,
                U(0),
                U(0),
                U(0),
                U(0),
                q,
                k,
                v,
                wt.ptr(block, 2),
                I(t),
                U(0),
                U(0),
                U(0),
                U(0),
                U(0),
                completion(mt * 16, true, 1)};
        add("vit", "joint_qkv_completion", {mt * 16}, {32, 4}, std::move(args), true);
        add("vit", "joint_attention_completion", {32, (t + 255) / 256}, {32, 4},
            {q, k, v, U(0), at, U(0), U(0), I(t), I(32), I(1), U(0), U(0), U(0), U(0),
             completion(mt * 32)},
            true);
        args = matrix_args(at, wt.ptr(block, 4), t, 1024, 1024, 2, 4, 0, 0, partial, x0, 0x100000);
        args.push_back(I(32));
        add("vit", "matrix_projection", {mt * 8, 1, 4}, {32, 4}, std::move(args));
        flat("bridge", "publish_split", t * 1024, {partial, scratch, hp, I(t * 1024), I(4)});
        flat("bridge", "merge_phase_completion", t * 1024,
             {partial, scratch, hp, U(0), I(t * 1024), I(4), U(0), I(0), x1,
              U(block == 38 ? pcm : nextpm), completion(mt * 8, true, 3, block == 38)},
             true);
        current = x1;
    }
    auto r99 = repack99(ph, pw);
    flat("bridge", "permute", t * 1024, {current, x0, map(r99), I(t * 1024)});
    U up_half = buffer(t * 512 * 2), upmap = map(halfmap(pool));
    auto inverse_pc = gather(argsort(r99), pc);
    add("bridge", "matrix_phase", {mt, 4, 4}, {128},
        matrix_args(x0, wt.ptr(39, 0), t, 1024, 512, 2, 4, 0, 0, partial, 0, 0, &inverse_pc));
    flat("bridge", "publish_split", t * 512, {partial, scratch, upmap, I(t * 512), I(4)});
    flat("bridge", "merge_phase", t * 512,
         {partial, scratch, upmap, up_half, I(t * 512), I(4), U(0), I(0), U(0), U(0)});
    Vec ep(rows * 512);
    for (int r = 0; r < rows; ++r)
    {
        int within = r % (h * 4), slot = within % 32;
        int y = (within / 32) * 8 + ((slot >> 3) & 1) + ((slot & 1) << 1) +
                (((slot >> 4) & 1) << 2),
            x = (r / (h * 4)) * 4 + ((slot >> 1) & 3);
        for (int n = 0; n < 512; ++n)
            ep[r * 512 + n] = (y * w + x) * 512 + n;
    }
    flat("bridge", "up_exit_completion", rows * 512,
         {up_half, skip56, U(wt.ptr(39, 0) + 0x80000), b, map(ep), map(decmap), I(rows * 512), I(w),
          I(pw), completion(sh * sw * 2, true)},
         true);
    current = b;
    for (int block = 40; block < 48; ++block)
        current = swin(block, current);
}
} // namespace dlss5::nr::detail

namespace dlss5::nr
{
struct Engine::Impl
{
    CUcontext context = nullptr;
    CUdevice device = 0;
    bool retained = false;
    std::unique_ptr<detail::Compiler> compiler;
    std::unique_ptr<detail::Weights> weights;
    std::unique_ptr<detail::State> state;
    Impl(const std::filesystem::path &dir, int ordinal)
    {
        using namespace detail;
        check(cuInit(0));
        check(cuDeviceGet(&device, ordinal));
        check(cuCtxGetCurrent(&context));
        if (!context)
            throw std::runtime_error("NR requires caller's current primary CUDA context");
        CUdevice actual;
        check(cuCtxGetDevice(&actual));
        if (actual != device)
            throw std::runtime_error("NR device differs from current context");
        int major, minor;
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
        check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
        if (major != 8 || minor != 9)
            throw std::runtime_error(
                "Native NR selected kernels require SM89; no automatic fallback");
        CUcontext primary;
        check(cuDevicePrimaryCtxRetain(&primary, device));
        retained = true;
        try
        {
            if (primary != context)
                throw std::runtime_error("NR requires primary CUDA context, not a private context");
            compiler = std::make_unique<Compiler>();
            weights = std::make_unique<Weights>(context, dir);
        }
        catch (...)
        {
            weights.reset();
            compiler.reset();
            cuDevicePrimaryCtxRelease(device);
            retained = false;
            throw;
        }
    }
    ~Impl()
    {
        if (retained)
        {
            CUcontext old = nullptr;
            cuCtxGetCurrent(&old);
            cuCtxSetCurrent(context);
            state.reset();
            weights.reset();
            compiler.reset();
            cuCtxSetCurrent(old);
            cuDevicePrimaryCtxRelease(device);
        }
    }
    void guard() const
    {
        CUcontext c;
        detail::check(cuCtxGetCurrent(&c));
        if (c != context)
            throw std::runtime_error("NR cannot execute in a different CUDA context");
    }
    void pointer(CUdeviceptr p, size_t size) const
    {
        if (!p || p % 16)
            throw std::invalid_argument("NR buffers must be non-null and 16-byte aligned");
        CUcontext owner;
        detail::check(cuPointerGetAttribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, p));
        if (owner != context)
            throw std::invalid_argument("NR buffer belongs to a different CUDA context");
        CUdeviceptr base;
        size_t bytes;
        detail::check(cuMemGetAddressRange(&base, &bytes, p));
        if (p < base || p - base > bytes || size > bytes - size_t(p - base))
            throw std::invalid_argument("NR buffer allocation is too small");
    }
};
Engine::Engine(const std::filesystem::path &dir, int device)
    : impl_(std::make_unique<Impl>(dir, device))
{
}
Engine::~Engine() = default;
void Engine::prepare(int h, int w)
{
    impl_->guard();
    detail::Shape shape(h, w);
    if (impl_->state && impl_->state->shape.h == h && impl_->state->shape.w == w)
        return;
    auto state =
        std::make_unique<detail::State>(shape, impl_->context, *impl_->compiler, *impl_->weights);
    impl_->state = std::move(state);
}
void Engine::infer(CUdeviceptr packet, CUdeviceptr head, int h, int w, CUstream stream)
{
    using namespace detail;
    impl_->guard();
    auto *s = impl_->state.get();
    if (!s || s->shape.h != h || s->shape.w != w)
        throw std::logic_error("NR infer requires matching prepare(h,w)");
    CUcontext stream_context;
    if (stream)
    {
        check(cuStreamGetCtx(stream, &stream_context));
        if (stream_context != impl_->context)
            throw std::invalid_argument("NR stream belongs to a different CUDA context");
    }
    CUstreamCaptureStatus capture;
    check(cuStreamIsCapturing(stream, &capture));
    if (capture != CU_STREAM_CAPTURE_STATUS_NONE)
        throw std::invalid_argument("NR status readback is not graph-capturable");
    size_t input_bytes = size_t(h) * w * 16 * 4, output_bytes = size_t(h) * w * 4 * 4;
    impl_->pointer(packet, input_bytes);
    impl_->pointer(head, output_bytes);
    if ((head >= packet && head - packet < input_bytes) ||
        (packet >= head && packet - head < output_bytes))
        throw std::invalid_argument("NR packet/head must not overlap");
    s->pre.args[1].set(U(packet));
    s->post.args[6].set(U(head));
    try
    {
        s->clear.launch(stream);
        s->pre.launch(stream);
        for (auto &step : s->enc)
            step.launch(stream);
        check(cuMemsetD32Async(s->completion_slab, 0xffffffff, s->completion_count, stream));
        for (auto &step : s->deep)
            step.launch(stream);
        for (auto &step : s->dec)
            step.launch(stream);
        s->post.launch(stream);
        check(cuMemcpyDtoHAsync(s->host_status, s->status, 4, stream));
        check(cuStreamSynchronize(stream));
    }
    catch (...)
    {
        cuStreamSynchronize(stream);
        throw;
    }
    if (*s->host_status)
        throw std::runtime_error("NR packet/head validation failed (1=nonfinite/overflowing Half "
                                 "input, 4=nonfinite head): " +
                                 std::to_string(*s->host_status));
}
} // namespace dlss5::nr
