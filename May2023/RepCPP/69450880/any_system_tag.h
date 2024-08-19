

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>

namespace hydra_thrust
{

struct any_system_tag
: hydra_thrust::execution_policy<any_system_tag>
{
template<typename T> operator T () const {return T();}
};

typedef HYDRA_THRUST_DEPRECATED any_system_tag any_space_tag;

} 

