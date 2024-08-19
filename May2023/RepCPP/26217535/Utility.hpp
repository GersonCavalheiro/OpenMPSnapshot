
#pragma once

#include <alpaka/core/Common.hpp>

#if BOOST_LANG_CUDA && BOOST_COMP_CLANG_CUDA || BOOST_COMP_HIP
#    include <type_traits>
#else
#    include <utility>
#endif

namespace alpaka::core
{
#if BOOST_LANG_CUDA && BOOST_COMP_CLANG_CUDA || BOOST_COMP_HIP
template<class T>
ALPAKA_FN_HOST_ACC std::add_rvalue_reference_t<T> declval();
#else
using std::declval;
#endif

template<typename Integral>
[[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto divCeil(Integral a, Integral b) -> Integral
{
return (a + b - 1) / b;
}

template<typename Integral>
[[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto intPow(Integral base, Integral n) -> Integral
{
if(n == 0)
return 1;
auto r = base;
for(Integral i = 1; i < n; i++)
r *= base;
return r;
}

template<typename Integral>
[[nodiscard]] ALPAKA_FN_HOST_ACC constexpr auto nthRootFloor(Integral value, Integral n) -> Integral
{
Integral L = 0;
Integral R = value + 1;
while(L != R - 1)
{
Integral const M = (L + R) / 2;
if(intPow(M, n) <= value)
L = M;
else
R = M;
}
return L;
}
} 
