

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#define __HYDRA_THRUST_DEVICE_SYSTEM_MEMORY_HEADER <__HYDRA_THRUST_DEVICE_SYSTEM_ROOT/memory_resource.h>
#include __HYDRA_THRUST_DEVICE_SYSTEM_MEMORY_HEADER
#undef __HYDRA_THRUST_DEVICE_SYSTEM_MEMORY_HEADER

namespace hydra_thrust
{


typedef hydra_thrust::system::__HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE::memory_resource
device_memory_resource;
typedef hydra_thrust::system::__HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE::universal_memory_resource
universal_memory_resource;
typedef hydra_thrust::system::__HYDRA_THRUST_DEVICE_SYSTEM_NAMESPACE::universal_host_pinned_memory_resource
universal_host_pinned_memory_resource;


} 

