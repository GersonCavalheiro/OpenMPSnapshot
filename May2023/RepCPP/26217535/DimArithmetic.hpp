

#pragma once

#include <alpaka/dim/DimIntegralConst.hpp>

#include <type_traits>

namespace alpaka::trait
{
template<typename T>
struct DimType<T, std::enable_if_t<std::is_arithmetic_v<T>>>
{
using type = DimInt<1u>;
};
} 
