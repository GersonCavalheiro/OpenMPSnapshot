

#pragma once

#include <alpaka/dim/Traits.hpp>

#include <type_traits>

namespace alpaka
{
template<std::size_t N>
using DimInt = std::integral_constant<std::size_t, N>;
} 
