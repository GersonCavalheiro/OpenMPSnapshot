




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>

namespace hydra_thrust
{




template<typename T>
inline void device_delete(hydra_thrust::device_ptr<T> ptr,
const size_t n = 1);



} 

#include <hydra/detail/external/hydra_thrust/detail/device_delete.inl>

