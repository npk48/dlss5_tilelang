// Functional NR kernel group. Internal namespaces preserve independent variants.
#include <cuda_runtime.h>

// ============================================================================
// OUTER utility.cu
// Isolated implementation; exported CUDA entry names and parameter ABI retained.
// ============================================================================
namespace nr_outer_utility_11 {
extern "C" __global__ void clear_counters(int *counters, int n, int *status) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        counters[i] = -1;
    if (i == 0)
        *status = 0;
}
extern "C" __global__ void pool8_padding(unsigned char *r, int offset, int h, int w, int ph,
                                         int pw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ph * pw * 512)
        return;
    int plane = i / (ph * pw * 16), z = i % (ph * pw * 16), pixel = z / 16;
    int y = pixel / pw, x = pixel % pw;
    if (y * 2 >= h || x * 2 >= w)
        r[offset + i] = 0;
}

} // namespace nr_outer_utility_11
