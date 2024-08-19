

#pragma once

#include <alpaka/meta/Integral.hpp>

#include <algorithm>
#include <limits>

namespace alpaka::core
{
template<typename T, typename V>
auto clipCast(V const& val) -> T
{
static_assert(
std::is_integral_v<T> && std::is_integral_v<V>,
"clipCast can not be called with non-integral types!");

auto constexpr max = static_cast<V>(std::numeric_limits<alpaka::meta::LowerMax<T, V>>::max());
auto constexpr min = static_cast<V>(std::numeric_limits<alpaka::meta::HigherMin<T, V>>::min());

return static_cast<T>(std::max(min, std::min(max, val)));
}
} 
