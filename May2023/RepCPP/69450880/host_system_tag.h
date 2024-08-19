

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#define __HYDRA_THRUST_HOST_SYSTEM_TAG_HEADER <__HYDRA_THRUST_HOST_SYSTEM_ROOT/detail/execution_policy.h>
#include __HYDRA_THRUST_HOST_SYSTEM_TAG_HEADER
#undef __HYDRA_THRUST_HOST_SYSTEM_TAG_HEADER

namespace hydra_thrust
{

typedef hydra_thrust::system::__HYDRA_THRUST_HOST_SYSTEM_NAMESPACE::tag host_system_tag;

} 

namespace hydra_thrust
{

typedef HYDRA_THRUST_DEPRECATED host_system_tag host_space_tag;

} 

