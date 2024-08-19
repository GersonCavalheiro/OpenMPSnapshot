

#ifndef LBT_CEM_CEIL
#define LBT_CEM_CEIL
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "is_inf.hpp"
#include "is_nan.hpp"
#include "is_almost_equal_eps_rel.hpp"


namespace lbt {
namespace cem {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr T ceil(T const x) noexcept {
if (cem::isNan(x) || cem::isPosInf(x) || cem::isNegInf(x)) {
return x;
}

return cem::isAlmostEqualEpsRel(static_cast<T>(static_cast<std::int64_t>(x)), x)
? static_cast<T>(static_cast<std::int64_t>(x))
: static_cast<T>(static_cast<std::int64_t>(x)) + ((x > 0) ? 1 : 0);
}

}
}

#endif 
