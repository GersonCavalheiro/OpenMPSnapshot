

#pragma once

#include <alpaka/core/Common.hpp>

namespace alpaka::rand
{
ALPAKA_FN_HOST_ACC constexpr static auto high32Bits(std::uint64_t const x) -> std::uint32_t
{
return static_cast<std::uint32_t>(x >> 32);
}

ALPAKA_FN_HOST_ACC constexpr static auto low32Bits(std::uint64_t const x) -> std::uint32_t
{
return static_cast<std::uint32_t>(x & 0xffffffff);
}


ALPAKA_FN_HOST_ACC constexpr static void multiplyAndSplit64to32(
std::uint64_t const a,
std::uint64_t const b,
std::uint32_t& resultHigh,
std::uint32_t& resultLow)
{
std::uint64_t res64 = a * b;
resultHigh = high32Bits(res64);
resultLow = low32Bits(res64);
}
} 
