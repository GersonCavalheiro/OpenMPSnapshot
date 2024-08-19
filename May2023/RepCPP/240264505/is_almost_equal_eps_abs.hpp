

#ifndef LBT_CEM_IS_ALMOST_EQUAL_EPS_ABS
#define LBT_CEM_IS_ALMOST_EQUAL_EPS_ABS
#pragma once

#include <limits>
#include <type_traits>

#include "abs.hpp"


namespace lbt {
namespace cem {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isAlmostEqualEpsAbs(T const a, T const b, T epsilon = 10*std::numeric_limits<T>::epsilon()) noexcept {
return (cem::abs(a - b) < epsilon);
}

}
}

#endif 
