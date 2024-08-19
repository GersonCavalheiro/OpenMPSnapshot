
#pragma once

#include <limits>

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

HYDRA_THRUST_BEGIN_NS

template <typename T>
struct numeric_limits : std::numeric_limits<T> {};

HYDRA_THRUST_END_NS

