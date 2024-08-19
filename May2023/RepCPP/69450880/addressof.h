
#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#  include <memory>
#endif

HYDRA_THRUST_BEGIN_NS



template <typename T>
__host__ __device__
T* addressof(T& arg) 
{
return reinterpret_cast<T*>(
&const_cast<char&>(reinterpret_cast<const volatile char&>(arg))
);
}


HYDRA_THRUST_END_NS

