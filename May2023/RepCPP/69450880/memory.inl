

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/memory.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/malloc_and_free.h>
#include <limits>

namespace hydra_thrust
{
namespace cuda_cub
{

__host__ __device__
pointer<void> malloc(std::size_t n)
{
tag cuda_tag;
return pointer<void>(hydra_thrust::cuda_cub::malloc(cuda_tag, n));
} 

template<typename T>
__host__ __device__
pointer<T> malloc(std::size_t n)
{
pointer<void> raw_ptr = hydra_thrust::cuda_cub::malloc(sizeof(T) * n);
return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} 

__host__ __device__
void free(pointer<void> ptr)
{
tag cuda_tag;
return hydra_thrust::cuda_cub::free(cuda_tag, ptr.get());
} 

} 
} 

