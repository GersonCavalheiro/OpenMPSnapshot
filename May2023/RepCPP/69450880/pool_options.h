



#pragma once

#include <cstddef>

#include <hydra/detail/external/hydra_thrust/detail/integer_math.h>

#include <hydra/detail/external/hydra_thrust/mr/detail/config.h>

namespace hydra_thrust
{
namespace mr
{




struct pool_options
{

std::size_t min_blocks_per_chunk;

std::size_t min_bytes_per_chunk;

std::size_t max_blocks_per_chunk;

std::size_t max_bytes_per_chunk;


std::size_t smallest_block_size;

std::size_t largest_block_size;


std::size_t alignment;


bool cache_oversized;


std::size_t cached_size_cutoff_factor;

std::size_t cached_alignment_cutoff_factor;


bool validate() const
{
if (!detail::is_power_of_2(smallest_block_size)) return false;
if (!detail::is_power_of_2(largest_block_size)) return false;
if (!detail::is_power_of_2(alignment)) return false;

if (max_bytes_per_chunk == 0 || max_blocks_per_chunk == 0) return false;
if (smallest_block_size == 0 || largest_block_size == 0) return false;

if (min_blocks_per_chunk > max_blocks_per_chunk) return false;
if (min_bytes_per_chunk > max_bytes_per_chunk) return false;

if (smallest_block_size > largest_block_size) return false;

if (min_blocks_per_chunk * smallest_block_size > max_bytes_per_chunk) return false;
if (min_blocks_per_chunk * largest_block_size > max_bytes_per_chunk) return false;

if (max_blocks_per_chunk * largest_block_size < min_bytes_per_chunk) return false;
if (max_blocks_per_chunk * smallest_block_size < min_bytes_per_chunk) return false;

if (alignment > smallest_block_size) return false;

return true;
}
};



} 
} 

