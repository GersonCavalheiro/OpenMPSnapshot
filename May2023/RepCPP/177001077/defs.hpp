
#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

#ifndef MARLIN_N_DIMS
#    define MARLIN_N_DIMS 3
#endif

namespace marlin
{
constexpr auto dim = MARLIN_N_DIMS;
static_assert(std::is_integral<decltype(dim)>::value,
"Number of dimensions must be a positive integer.");
static_assert(dim > 1, "The number of dimensions must be at least two.");

constexpr auto n_sweeps = 1 << dim;
constexpr auto n_boundaries = 2 * dim;

using index_t = std::size_t;
using point_t = std::array<index_t, dim>;

using scalar_t = double;
using vector_t = std::array<scalar_t, dim>;
}    
