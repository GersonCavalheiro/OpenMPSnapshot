

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/cuda/detail/util.h>
#include <cstdio>

namespace hydra_thrust
{
namespace system
{
namespace cuda
{
namespace detail
{


inline __device__
void terminate()
{
hydra_thrust::cuda_cub::terminate();
}


inline __host__ __device__
void terminate_with_message(const char* message)
{
printf("%s\n", message);
hydra_thrust::cuda_cub::terminate();
}


} 
} 
} 
} 

