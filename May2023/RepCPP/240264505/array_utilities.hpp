

#ifndef LBT_ARRAY_UTILITIES
#define LBT_ARRAY_UTILITIES
#pragma once

#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>

#include "constexpr_math/constexpr_math.hpp"
#include "general/type_definitions.hpp"


namespace lbt {
namespace test {


template <typename T, std::size_t N, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isSymmetric(lbt::StackArray<T,N> const& arr) noexcept {
constexpr std::size_t hspeed {N/2};
bool is_symmetric {true};
is_symmetric &= lbt::cem::isAlmostEqualEpsAbs(arr.at(0), arr.at(hspeed));
for (std::size_t i = 1; i < hspeed; ++i) {
is_symmetric &= lbt::cem::isAlmostEqualEpsAbs(arr.at(i), arr.at(i + hspeed));
}
return is_symmetric;
}


template <typename T, std::size_t N, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool isAntimetric(lbt::StackArray<T,N> const& arr) noexcept {
constexpr std::size_t hspeed {N/2};
bool is_antimetric {true};
is_antimetric &= lbt::cem::isAlmostEqualEpsAbs(arr.at(0), arr.at(hspeed));
for (std::size_t i = 1; i < hspeed; ++i) {
is_antimetric &= lbt::cem::isAlmostEqualEpsAbs(arr.at(i), -arr.at(i + hspeed));
}
return is_antimetric;
}


template <typename T, std::size_t N, typename std::enable_if_t<std::is_floating_point_v<T>>* = nullptr>
constexpr bool sumsTo(lbt::StackArray<T,N> const& arr, T const expected_sum) noexcept {
T const sum = std::accumulate(std::begin(arr), std::end(arr), static_cast<T>(0));
return lbt::cem::isAlmostEqualEpsAbs(sum, expected_sum);
}


constexpr bool isAligned(void const* const ptr, std::size_t const alignment) noexcept {
return (reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0);
}

}
}

#endif 
