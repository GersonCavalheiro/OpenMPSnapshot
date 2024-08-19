




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <cstddef>
#include <hydra/detail/external/hydra_thrust/device_ptr.h>

namespace hydra_thrust
{




template <typename T>
device_ptr<T> device_new(device_ptr<void> p,
const size_t n = 1);


template <typename T>
device_ptr<T> device_new(device_ptr<void> p,
const T &exemplar,
const size_t n = 1);


template <typename T>
device_ptr<T> device_new(const size_t n = 1);



} 

#include <hydra/detail/external/hydra_thrust/detail/device_new.inl>

