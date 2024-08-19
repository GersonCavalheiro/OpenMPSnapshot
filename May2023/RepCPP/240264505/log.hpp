

#ifndef LBT_CEM_LOG
#define LBT_CEM_LOG
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "definitions.hpp"
#include "exp.hpp"
#include "is_inf.hpp"
#include "is_nan.hpp"
#include "is_almost_equal_eps_rel.hpp"
#include "mathematical_constants.hpp"


namespace lbt {
namespace cem {

namespace detail {

template <typename T, std::int64_t RD = cem::default_max_recursion_depth, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr T logNewton(T const x, T const prev, std::int64_t const depth = 0) noexcept {
if (depth >= RD) {
return prev;
}
auto const curr = prev + static_cast<T>(2.0)*(x-cem::exp(prev))/(x+cem::exp(prev));
return cem::isAlmostEqualEpsRel(prev, curr) ? curr : logNewton(x, curr, depth+1);
}
}


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr T log(T const x) noexcept {
if (cem::isAlmostEqualEpsRel<T>(x, static_cast<T>(0.0))) {
return -std::numeric_limits<T>::infinity();
} else if (cem::isAlmostEqualEpsRel<T>(x, static_cast<T>(1.0))) {
return static_cast<T>(0.0);
} else if (x < static_cast<T>(0.0)) {
return std::numeric_limits<T>::quiet_NaN();
} else if (cem::isNegInf(x)) {
return std::numeric_limits<T>::quiet_NaN();
} else if (cem::isPosInf(x)) {
return std::numeric_limits<T>::infinity();
} else if (cem::isNan(x)) {
return std::numeric_limits<T>::quiet_NaN();
} else if (cem::isAlmostEqualEpsRel<T>(x, cem::e<T>)) {
return static_cast<T>(1.0);
}

return cem::detail::logNewton(x, static_cast<T>(0.0), 0);
}

}
}

#endif 
