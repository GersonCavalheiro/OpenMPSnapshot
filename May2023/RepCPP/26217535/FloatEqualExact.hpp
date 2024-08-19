

#pragma once

#include <alpaka/core/Common.hpp>

#include <type_traits>

namespace alpaka
{
namespace math
{

template<typename T>
ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto floatEqualExactNoWarning(T a, T b) -> bool
{
static_assert(std::is_floating_point_v<T>, "floatEqualExactNoWarning is for floating point values only!");

#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
return a == b;
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
}
} 
} 
