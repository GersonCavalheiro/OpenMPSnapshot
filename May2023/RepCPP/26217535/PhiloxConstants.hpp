

#pragma once

#include <alpaka/rand/Philox/MultiplyAndSplit64to32.hpp>

#include <cstdint>
#include <utility>


namespace alpaka::rand::engine
{

template<typename TParams>
class PhiloxConstants
{
public:
static constexpr std::uint64_t WEYL_64_0()
{
return 0x9E3779B97F4A7C15; 
}
static constexpr std::uint64_t WEYL_64_1()
{
return 0xBB67AE8584CAA73B; 
}

static constexpr std::uint32_t WEYL_32_0()
{
return high32Bits(WEYL_64_0()); 
}
static constexpr std::uint32_t WEYL_32_1()
{
return high32Bits(WEYL_64_1()); 
}

static constexpr std::uint32_t MULTIPLITER_4x32_0()
{
return 0xCD9E8D57; 
}
static constexpr std::uint32_t MULTIPLITER_4x32_1()
{
return 0xD2511F53; 
}
};
} 
