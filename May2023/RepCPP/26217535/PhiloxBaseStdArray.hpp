

#pragma once

#include <array>
#include <cstdint>

namespace alpaka::rand::engine
{

template<typename TParams>
class PhiloxBaseStdArray
{
public:
using Counter = std::array<std::uint32_t, TParams::counterSize>; 
using Key = std::array<std::uint32_t, TParams::counterSize / 2>; 
template<typename TScalar>
using ResultContainer
= std::array<TScalar, TParams::counterSize>; 
};
} 
