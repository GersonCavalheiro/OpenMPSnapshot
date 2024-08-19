

#pragma once

#include <cstdint>
#include <tuple>

namespace alpaka::test
{
using TestIdxs = std::tuple<
#if !defined(ALPAKA_CI)
std::int64_t,
#endif
std::uint64_t,
std::int32_t
#if !defined(ALPAKA_CI)
,
std::uint32_t
#endif
>;
} 
