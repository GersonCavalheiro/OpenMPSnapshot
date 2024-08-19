

#ifndef LBT_CEM_IS_ALMOST_EQUAL_EPS_REL
#define LBT_CEM_IS_ALMOST_EQUAL_EPS_REL
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "abs.hpp"


namespace lbt {
namespace cem {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isAlmostEqualEpsRel(T const a, T const b, std::uint8_t const max_distance = 4) noexcept {
auto const diff {cem::abs(a-b)};
auto const sum {cem::abs(a+b)};
constexpr auto max {std::numeric_limits<T>::max()};
auto const norm {sum < max ? sum : max};
return diff <= std::numeric_limits<T>::epsilon()*norm*max_distance || 
diff < std::numeric_limits<T>::min();
}

}
}

#endif 
