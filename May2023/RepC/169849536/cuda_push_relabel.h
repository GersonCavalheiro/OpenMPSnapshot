#pragma once
int push_relabel(int blocks_per_grid, int threads_per_block, int N, int src, int sink, int *cap, int *flow);
#define GPUErrChk(ans) { utils::GPUAssert((ans), __FILE__, __LINE__); }
namespace utils {
int idx(int x, int y, int n);
inline void GPUAssert(cudaError_t code, const char *file, int line, bool abort = true) {
if (code != cudaSuccess) {
fprintf(stderr, "GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
if (abort)
exit(code);
}
}
inline __device__ int dev_idx(int x, int y, int n) {
return x * n + y;
}
}
