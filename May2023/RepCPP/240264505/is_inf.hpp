

#ifndef LBT_CEM_IS_INF
#define LBT_CEM_IS_INF
#pragma once

#include <type_traits>


namespace lbt {
namespace cem {


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isPosInf(T const x) noexcept {
return (x > 0 && x/x != x/x);
}


template <typename T, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isNegInf(T const x) noexcept {
return (x < 0 && x/x != x/x);
}

}
}

#endif 
