#pragma once

#include <iostream>
#include <string>
using std::string;

#include <cuda_runtime.h>

#define PARACABS_DEBUG true

#if PARACABS_DEBUG
#define handle_cuda_error(body)                                                  \
{                                                                                \
cudaError_t err = body;                                                      \
if (err!=cudaSuccess) printf ("CUDA ERROR : %s\n", cudaGetErrorString(err)); \
}
#else
#define handle_cuda_error(body) body
#endif


#define accel  __host__ __device__


namespace paracabs
{
namespace accelerator
{
inline size_t& nblocks()
{
static size_t value = 1;
return value;
}

inline size_t& nthreads()
{
static size_t value = 1;
return value;
}


inline unsigned int nGPUs ()
{
int n;
handle_cuda_error (cudaGetDeviceCount (&n));
return n;
}


inline string get_gpu_name (const int i)
{
cudaDeviceProp prop;
handle_cuda_error (cudaGetDeviceProperties (&prop, i));
return prop.name;
}


inline void list_accelerators ()
{
for (unsigned int i = 0; i < nGPUs (); i++)
{
std::cout << get_gpu_name (i) << std::endl;
}
}


inline void synchronize ()
{
handle_cuda_error (cudaDeviceSynchronize ());
}


inline void* malloc (const size_t num)
{
void* ptr;
handle_cuda_error (cudaMalloc (&ptr, num));
handle_cuda_error (cudaDeviceSynchronize ());
return ptr;
}


inline void free (void* ptr)
{
handle_cuda_error (cudaDeviceSynchronize ());
handle_cuda_error (cudaFree (ptr));
}

inline void memcpy_to_accelerator (void* dst, const void* src, const size_t size)
{
handle_cuda_error (cudaMemcpy (dst, src, size, cudaMemcpyHostToDevice));
}

inline void memcpy_from_accelerator (void* dst, const void* src, const size_t size)
{
handle_cuda_error (cudaMemcpy (dst, src, size, cudaMemcpyDeviceToHost));
}

struct AcceleratorThreads
{
__device__ inline size_t thread_id ()
{
return blockIdx.x * blockDim.x + threadIdx.x;
}

__host__   inline size_t tot_nthreads ()
{
return nblocks() * nthreads();
}
};
}
}



#define accelerated_for(i, total, ... )                                       \
{							                                                  \
copyContextAccelerator() = true;                                          \
auto lambda = [=, *this] __device__ (size_t i) mutable                    \
{		                                                                  \
__VA_ARGS__;							                              \
};									                                      \
decltype(lambda)* lambda_ptr =                                            \
(decltype(lambda)*) paracabs::accelerator::malloc (sizeof(lambda));   \
paracabs::accelerator::memcpy_to_accelerator                              \
(lambda_ptr, &lambda, sizeof(lambda));                                \
dim3 cuda_nblocks  = paracabs::accelerator::nblocks();                    \
dim3 cuda_nthreads = paracabs::accelerator::nthreads();                   \
apply_lambda <<<cuda_nblocks, cuda_nthreads>>> (total, lambda_ptr);       \
copyContextAccelerator() = false;                                         \
}


#define accelerated_for_outside_class(i, total, ... )                         \
{							                                                  \
copyContextAccelerator() = true;                                          \
auto lambda = [=] __device__ (size_t i) mutable                           \
{		                                                                  \
__VA_ARGS__;							                              \
};									                                      \
decltype(lambda)* lambda_ptr =                                            \
(decltype(lambda)*) paracabs::accelerator::malloc (sizeof(lambda));   \
paracabs::accelerator::memcpy_to_accelerator                              \
(lambda_ptr, &lambda, sizeof(lambda));                                \
dim3 cuda_nblocks  = paracabs::accelerator::nblocks();                    \
dim3 cuda_nthreads = paracabs::accelerator::nthreads();                   \
apply_lambda <<<cuda_nblocks, cuda_nthreads>>> (total, lambda_ptr);       \
copyContextAccelerator() = false;                                         \
}


template <typename Lambda>
__global__ void apply_lambda (const size_t stop, Lambda* lambda)
{
const size_t start  = blockIdx.x * blockDim.x + threadIdx.x;
const size_t stride =  gridDim.x * blockDim.x;

for (size_t i = start; i < stop; i += stride)
{
lambda->operator()(i);
}
}
