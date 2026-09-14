#include "nr_engine.h"
#include "nr_layouts.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
using namespace dlss5::nr;
namespace
{
void ck(CUresult c)
{
    if (c)
    {
        const char *t = nullptr;
        cuGetErrorString(c, &t);
        throw std::runtime_error(t ? t : "CUDA error");
    }
}
template <class T> void write(const std::filesystem::path &p, const std::vector<T> &v)
{
    std::ofstream f(p, std::ios::binary);
    if (!f.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(T)))
        throw std::runtime_error("output write failed");
}
void text(const std::filesystem::path &p, const std::string &s)
{
    std::ofstream f(p, std::ios::binary);
    f << s;
}
std::vector<float> read(const char *file, size_t count)
{
    std::ifstream f(file, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(count * 4))
        throw std::runtime_error("packet file must contain exactly BCHW16 float32");
    std::vector<float> v(count);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), count * 4);
    return v;
}
void dump(int h, int w, const std::filesystem::path &dir)
{
    using namespace detail;
    std::filesystem::create_directories(dir);
    Shape s(h, w);
    std::map<int, Pool> ds;
    for (auto v : {std::pair<int, int>{2, 8}, {4, 14}, {8, 22}})
    {
        auto p = pool_metadata(s.outer(v.second));
        write(dir / ("pool" + std::to_string(v.first) + "_input.i32"), p.input);
        write(dir / ("pool" + std::to_string(v.first) + "_output.i32"), p.output);
        ds.emplace(v.first, std::move(p));
    }
    for (auto v : {std::pair<int, int>{2, 62}, {4, 56}, {8, 48}})
    {
        auto p = up_metadata(s.outer(v.second));
        write(dir / ("up" + std::to_string(v.first) + "_input.i32"), p.input);
        write(dir / ("up" + std::to_string(v.first) + "_inverse.i32"), p.inverse);
        if (v.first != 2)
            text(dir / ("routes" + std::to_string(v.first) + ".cuh"),
                 routes(v.first, ds.at(v.first), p));
    }
    text(dir / "addresses.cuh", addresses(s, ds));
    int ph = align(s.dh, 8) / 2, pw = align(s.dw, 8) / 2;
    auto pc = pointmap(ph, pw, 1024), ff = pointmap(ph, pw, 4096),
         pool = raw16(ph, pw, false, true);
    write(dir / "enc.i32", raw16(s.dh, s.dw));
    write(dir / "dec.i32", raw16(s.dh, s.dw, true));
    write(dir / "pc.i32", pc);
    write(dir / "pi.i32", pointmap(ph, pw, 1024, true));
    write(dir / "ff.i32", ff);
    write(dir / "pool.i32", pool);
    write(dir / "terminal.i32", raw16(ph, pw, false, true, 1024));
    write(dir / "half.i32", halfmap(pc));
    write(dir / "repack99.i32", repack99(ph, pw));
    for (int i = 0; i < 4; ++i)
    {
        auto p = local_packets(s.dh, s.dw, i);
        write(dir / ("local" + std::to_string(i) + ".i32"), p.plan);
        write(dir / ("members" + std::to_string(i) + ".i32"), p.members);
    }
    auto pk = packets(pool, ph * pw, 512);
    write(dir / "sources.i32", pk.sources);
    write(dir / "routes.u16", pk.routes);
}
struct DeviceBuffer
{
    CUdeviceptr p = 0;
    explicit DeviceBuffer(size_t n)
    {
        ck(cuMemAlloc(&p, n));
    }
    ~DeviceBuffer()
    {
        if (p)
            cuMemFree(p);
    }
};
} // namespace
int main(int argc, char **argv)
{
    try
    {
        if (argc == 5 && std::string(argv[1]) == "--dump-layouts")
        {
            dump(std::stoi(argv[2]), std::stoi(argv[3]), argv[4]);
            std::cout << "layout dump complete\n";
            return 0;
        }
        if (argc < 6 || (argc - 2) % 4)
        {
            std::cerr << "Usage: nr_smoke MODEL_DIR H W packet.f32 head.f32 [H W packet2.f32 "
                         "head2.f32 ...]\n";
            return 2;
        }
        ck(cuInit(0));
        CUdevice device;
        ck(cuDeviceGet(&device, 0));
        CUcontext context;
        ck(cuDevicePrimaryCtxRetain(&context, device));
        ck(cuCtxSetCurrent(context));
        CUstream stream;
        ck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        {
            Engine engine(argv[1], 0);
            bool tested_contracts = false;
            for (int i = 2; i < argc; i += 4)
            {
                int h = std::stoi(argv[i]), w = std::stoi(argv[i + 1]);
                auto packet = read(argv[i + 2], size_t(h) * w * 16);
                std::vector<float> head(size_t(h) * w * 4), repeat(head.size());
                DeviceBuffer input(packet.size() * 4), output(head.size() * 4),
                    other(head.size() * 4);
                auto start = std::chrono::steady_clock::now();
                std::cout << "prepare " << h << "x" << w << std::endl;
                engine.prepare(h, w);
                double prep =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                ck(cuMemcpyHtoDAsync(input.p, packet.data(), packet.size() * 4, stream));
                start = std::chrono::steady_clock::now();
                engine.infer(input.p, output.p, h, w, stream);
                double seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                ck(cuMemcpyDtoH(head.data(), output.p, head.size() * 4));
                engine.infer(input.p, other.p, h, w, stream);
                ck(cuMemcpyDtoH(repeat.data(), other.p, repeat.size() * 4));
                if (std::memcmp(head.data(), repeat.data(), head.size() * 4))
                    throw std::runtime_error("repeat head differs");
                for (float v : head)
                    if (!std::isfinite(v))
                        throw std::runtime_error("nonfinite native output");
                write(argv[i + 3], head);
                if (!tested_contracts)
                {
                    auto rejects = [&](auto f, const char *name) {
                        bool caught = false;
                        try
                        {
                            f();
                        }
                        catch (const std::exception &)
                        {
                            caught = true;
                        }
                        if (!caught)
                            throw std::runtime_error(std::string("missing validation: ") + name);
                    };
                    rejects([&] { engine.prepare(65, 64); }, "shape");
                    rejects([&] { engine.infer(input.p, output.p, h + 64, w, stream); },
                            "unprepared shape");
                    rejects([&] { engine.infer(input.p, input.p, h, w, stream); }, "overlap");
                    ck(cuCtxSetCurrent(nullptr));
                    rejects([&] { engine.prepare(h, w); }, "wrong current context");
                    ck(cuCtxSetCurrent(context));
                    float invalid = std::numeric_limits<float>::infinity();
                    ck(cuMemcpyHtoD(input.p, &invalid, 4));
                    rejects([&] { engine.infer(input.p, other.p, h, w, stream); },
                            "nonfinite status");
                    ck(cuMemcpyHtoD(input.p, packet.data(), packet.size() * 4));
                    engine.infer(input.p, other.p, h, w, stream);
                    ck(cuMemcpyDtoH(repeat.data(), other.p, repeat.size() * 4));
                    if (std::memcmp(head.data(), repeat.data(), head.size() * 4))
                        throw std::runtime_error("recovery after invalid packet differs");
                    tested_contracts = true;
                    std::cout << "contracts: shape, prepare, overlap, context, nonfinite status, "
                                 "recovery PASS\n";
                }
                std::cout << "native " << h << "x" << w << " prepare_s=" << prep
                          << " infer_s=" << seconds << " repeat_bitwise=PASS head=" << argv[i + 3]
                          << std::endl;
            }
        }
        ck(cuStreamDestroy(stream));
        ck(cuDevicePrimaryCtxRelease(device));
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "nr_smoke: " << e.what() << std::endl;
        return 1;
    }
}
