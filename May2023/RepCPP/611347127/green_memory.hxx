#pragma once

#ifndef   HAS_NO_CUDA
#include <cuda.h> 

__host__ inline
void __cudaSafeCall(cudaError_t err, const char *file, const int line, char const *call) { 
if (cudaSuccess != err) {
std::fprintf(stdout, "[ERROR] CUDA call to %s at %s:%d\n%s\n", call, file, line, cudaGetErrorString(err));
std::fflush(stdout);
std::fprintf(stderr, "[ERROR] CUDA call to %s at %s:%d\n%s\n", call, file, line, cudaGetErrorString(err));
std::fflush(stderr);
exit(0);
}
} 
#define cuCheck(err) __cudaSafeCall((err), __FILE__, __LINE__, #err) 

#else  
#define __global__
#define __restrict__
#define __device__
#define __shared__
#define __unroll__
#define __host__

struct dim3 {
int x, y, z;
dim3(int xx, int yy=1, int zz=1) : x(xx), y(yy), z(zz) {}
}; 

#ifndef   HAS_TFQMRGPU
inline void __syncthreads(void) {} 
typedef int cudaError_t;
inline cudaError_t cudaDeviceSynchronize(void) { return 0; } 
#else  
#define gpuStream_t cudaStream_t
#include "tfQMRgpu/include/tfqmrgpu_cudaStubs.hxx" 
#endif 

#endif 


template <typename T>
T* get_memory(size_t const size=1, int const echo=0, char const *const name="") {

#ifdef    DEBUG
if (echo > 0) {
size_t const total = size*sizeof(T);
std::printf("# managed memory: %lu x %.3f kByte = \t", size, sizeof(T)*1e-3);
if (total > 1e9) { std::printf("%.9f GByte\n", total*1e-9); } else 
if (total > 1e6) { std::printf("%.6f MByte\n", total*1e-6); } else 
{ std::printf("%.3f kByte\n", total*1e-3); }
} 
#endif 

T* ptr{nullptr};
#ifndef HAS_NO_CUDA
cuCheck(cudaMallocManaged(&ptr, size*sizeof(T)));
#else  
ptr = new T[size];
#endif 

#ifdef    DEBUGGPU
std::printf("# get_memory \t%lu x %.3f kByte = \t%.3f kByte, %s at %p\n", size, sizeof(T)*1e-3, size*sizeof(T)*1e-3, name, (void*)ptr);
#endif 

return ptr;
} 


template <typename T>
void _free_memory(T* & ptr, char const *const name="") {
if (nullptr != ptr) {
#ifdef    DEBUGGPU
std::printf("# free_memory %s at %p\n", name, (void*)ptr);
#endif 

#ifndef HAS_NO_CUDA
cuCheck(cudaFree((void*)ptr));
#else  
delete[] ptr;
#endif 
} 
ptr = nullptr;
} 

#define free_memory(PTR) _free_memory(PTR, #PTR)

template <typename real_t=float>
inline char const* real_t_name() { return (8 == sizeof(real_t)) ? "double" : "float"; }


