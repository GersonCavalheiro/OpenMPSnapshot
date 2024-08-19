

#ifndef LBT_CEM_ABS
#define LBT_CEM_ABS
#pragma once

#include <type_traits>

#include "is_inf.hpp"
#include "is_nan.hpp"


namespace lbt {
namespace cem {


template <typename T, typename std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
constexpr T abs(T const x) noexcept {
if constexpr (std::is_floating_point_v<T>) {
if (cem::isNan(x)) {
return x;
} else if (cem::isPosInf(x)) {
return x;
} else if (cem::isNegInf(x)) {
return -x;
}
}

return (x < 0) ? -x : x;
}

}
}

#endif 
