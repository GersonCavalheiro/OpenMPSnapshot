


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{

__hydra_thrust_exec_check_disable__
template<typename Assignable1, typename Assignable2>
__host__ __device__
inline void swap(Assignable1 &a, Assignable2 &b)
{
Assignable1 temp = a;
a = b;
b = temp;
} 

} 

