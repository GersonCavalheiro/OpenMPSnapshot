

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#define __HYDRA_THRUST_HOST_SYSTEM_MEMORY_HEADER <__HYDRA_THRUST_HOST_SYSTEM_ROOT/memory_resource.h>
#include __HYDRA_THRUST_HOST_SYSTEM_MEMORY_HEADER
#undef __HYDRA_THRUST_HOST_SYSTEM_MEMORY_HEADER

namespace hydra_thrust
{

typedef hydra_thrust::system::__HYDRA_THRUST_HOST_SYSTEM_NAMESPACE::memory_resource
host_memory_resource;

} 

