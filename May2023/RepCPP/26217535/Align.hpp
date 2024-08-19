

#pragma once

#include <alpaka/core/BoostPredef.hpp>

#include <cstddef>
#include <type_traits>

namespace alpaka::core
{
template<std::size_t N>
struct RoundUpToPowerOfTwo;

namespace detail
{
template<std::size_t N, bool TisPowerTwo>
struct RoundUpToPowerOfTwoHelper : std::integral_constant<std::size_t, N>
{
};
template<std::size_t N>
struct RoundUpToPowerOfTwoHelper<N, false>
: std::integral_constant<std::size_t, RoundUpToPowerOfTwo<(N | (N - 1)) + 1>::value>
{
};
} 
template<std::size_t N>
struct RoundUpToPowerOfTwo
: std::integral_constant<std::size_t, detail::RoundUpToPowerOfTwoHelper<N, (N & (N - 1)) == 0>::value>
{
};

namespace align
{
template<std::size_t TsizeBytes>
struct OptimalAlignment
: std::integral_constant<
std::size_t,
#if BOOST_COMP_GNUC
(TsizeBytes > 64) ? 128 :
#endif
(RoundUpToPowerOfTwo<TsizeBytes>::value)>
{
};
} 
} 

#define ALPAKA_OPTIMAL_ALIGNMENT(...)                                                                                 \
::alpaka::core::align::OptimalAlignment<sizeof(std::remove_cv_t<__VA_ARGS__>)>::value
