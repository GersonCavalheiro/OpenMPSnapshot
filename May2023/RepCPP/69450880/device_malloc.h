




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>
#include <cstddef> 

namespace hydra_thrust
{




inline hydra_thrust::device_ptr<void> device_malloc(const std::size_t n);


template<typename T>
inline hydra_thrust::device_ptr<T> device_malloc(const std::size_t n);



} 

#include <hydra/detail/external/hydra_thrust/detail/device_malloc.inl>

