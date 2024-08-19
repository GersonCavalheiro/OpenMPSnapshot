



#pragma once

#include <algorithm>

#include <hydra/detail/external/hydra_thrust/host_vector.h>
#include <hydra/detail/external/hydra_thrust/binary_search.h>
#include <hydra/detail/external/hydra_thrust/detail/seq.h>

#include <hydra/detail/external/hydra_thrust/mr/memory_resource.h>
#include <hydra/detail/external/hydra_thrust/mr/allocator.h>
#include <hydra/detail/external/hydra_thrust/mr/pool_options.h>

#include <cassert>

namespace hydra_thrust
{
namespace mr
{




template<typename Upstream, typename Bookkeeper>
class disjoint_unsynchronized_pool_resource HYDRA_THRUST_FINAL
: public memory_resource<typename Upstream::pointer>,
private validator2<Upstream, Bookkeeper>
{
public:

static pool_options get_default_options()
{
pool_options ret;

ret.min_blocks_per_chunk = 16;
ret.min_bytes_per_chunk = 1024;
ret.max_blocks_per_chunk = static_cast<std::size_t>(1) << 20;
ret.max_bytes_per_chunk = static_cast<std::size_t>(1) << 30;

ret.smallest_block_size = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT;
ret.largest_block_size = static_cast<std::size_t>(1) << 20;

ret.alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT;

ret.cache_oversized = true;

ret.cached_size_cutoff_factor = 16;
ret.cached_alignment_cutoff_factor = 16;

return ret;
}


disjoint_unsynchronized_pool_resource(Upstream * upstream, Bookkeeper * bookkeeper,
pool_options options = get_default_options())
: m_upstream(upstream),
m_bookkeeper(bookkeeper),
m_options(options),
m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
m_pools(m_bookkeeper),
m_allocated(m_bookkeeper),
m_cached_oversized(m_bookkeeper),
m_oversized(m_bookkeeper)
{
assert(m_options.validate());

pointer_vector free(m_bookkeeper);
pool p(free);
m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
}



disjoint_unsynchronized_pool_resource(pool_options options = get_default_options())
: m_upstream(get_global_resource<Upstream>()),
m_bookkeeper(get_global_resource<Bookkeeper>()),
m_options(options),
m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
m_pools(m_bookkeeper),
m_allocated(m_bookkeeper),
m_cached_oversized(m_bookkeeper),
m_oversized(m_bookkeeper)
{
assert(m_options.validate());

pointer_vector free(m_bookkeeper);
pool p(free);
m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
}


~disjoint_unsynchronized_pool_resource()
{
release();
}

private:
typedef typename Upstream::pointer void_ptr;
typedef typename hydra_thrust::detail::pointer_traits<void_ptr>::template rebind<char>::other char_ptr;

struct chunk_descriptor
{
std::size_t size;
void_ptr pointer;
};

typedef hydra_thrust::host_vector<
chunk_descriptor,
allocator<chunk_descriptor, Bookkeeper>
> chunk_vector;

struct oversized_block_descriptor
{
std::size_t size;
std::size_t alignment;
void_ptr pointer;

__host__ __device__
bool operator==(const oversized_block_descriptor & other) const
{
return size == other.size && alignment == other.alignment && pointer == other.pointer;
}

__host__ __device__
bool operator<(const oversized_block_descriptor & other) const
{
return size < other.size || (size == other.size && alignment < other.alignment);
}
};

struct equal_pointers
{
public:
__host__ __device__
equal_pointers(void_ptr p) : p(p)
{
}

__host__ __device__
bool operator()(const oversized_block_descriptor & desc) const
{
return desc.pointer == p;
}

private:
void_ptr p;
};

struct matching_alignment
{
public:
__host__ __device__
matching_alignment(std::size_t requested) : requested(requested)
{
}

__host__ __device__
bool operator()(const oversized_block_descriptor & desc) const
{
return desc.alignment >= requested;
}

private:
std::size_t requested;
};

typedef hydra_thrust::host_vector<
oversized_block_descriptor,
allocator<oversized_block_descriptor, Bookkeeper>
> oversized_block_vector;

typedef hydra_thrust::host_vector<
void_ptr,
allocator<void_ptr, Bookkeeper>
> pointer_vector;

struct pool
{
__host__
pool(const pointer_vector & free)
: free_blocks(free),
previous_allocated_count(0)
{
}

__host__
pool(const pool & other)
: free_blocks(other.free_blocks),
previous_allocated_count(other.previous_allocated_count)
{
}

__host__
~pool() {}

pointer_vector free_blocks;
std::size_t previous_allocated_count;
};

typedef hydra_thrust::host_vector<
pool,
allocator<pool, Bookkeeper>
> pool_vector;

Upstream * m_upstream;
Bookkeeper * m_bookkeeper;

pool_options m_options;
std::size_t m_smallest_block_log2;

pool_vector m_pools;
chunk_vector m_allocated;
oversized_block_vector m_cached_oversized;
oversized_block_vector m_oversized;

public:

void release()
{
for (std::size_t i = 0; i < m_pools.size(); ++i)
{
m_pools[i].free_blocks.clear();
m_pools[i].previous_allocated_count = 0;
}

for (std::size_t i = 0; i < m_allocated.size(); ++i)
{
m_upstream->do_deallocate(
m_allocated[i].pointer,
m_allocated[i].size,
m_options.alignment);
}

for (std::size_t i = 0; i < m_oversized.size(); ++i)
{
m_upstream->do_deallocate(
m_oversized[i].pointer,
m_oversized[i].size,
m_oversized[i].alignment);
}

m_allocated.clear();
m_oversized.clear();
m_cached_oversized.clear();
}

HYDRA_THRUST_NODISCARD virtual void_ptr do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
bytes = (std::max)(bytes, m_options.smallest_block_size);
assert(detail::is_power_of_2(alignment));

if (bytes > m_options.largest_block_size || alignment > m_options.alignment)
{
oversized_block_descriptor oversized;
oversized.size = bytes;
oversized.alignment = alignment;

if (m_options.cache_oversized && !m_cached_oversized.empty())
{
typename oversized_block_vector::iterator it = hydra_thrust::lower_bound(
hydra_thrust::seq,
m_cached_oversized.begin(),
m_cached_oversized.end(),
oversized);

if (it != m_cached_oversized.end())
{
std::size_t size_factor = (*it).size / bytes;
if (size_factor >= m_options.cached_size_cutoff_factor)
{
it = m_cached_oversized.end();
}
}

if (it != m_cached_oversized.end() && (*it).alignment < alignment)
{
it = find_if(it + 1, m_cached_oversized.end(), matching_alignment(alignment));
}

if (it != m_cached_oversized.end())
{
std::size_t alignment_factor = (*it).alignment / alignment;
if (alignment_factor >= m_options.cached_alignment_cutoff_factor)
{
it = m_cached_oversized.end();
}
}

if (it != m_cached_oversized.end())
{
oversized.pointer = (*it).pointer;
m_cached_oversized.erase(it);
return oversized.pointer;
}
}

oversized.pointer = m_upstream->do_allocate(bytes, alignment);
m_oversized.push_back(oversized);

return oversized.pointer;
}

std::size_t bytes_log2 = hydra_thrust::detail::log2_ri(bytes);
std::size_t bucket_idx = bytes_log2 - m_smallest_block_log2;
pool & bucket = m_pools[bucket_idx];

if (bucket.free_blocks.empty())
{
std::size_t bucket_size = static_cast<std::size_t>(1) << bytes_log2;

std::size_t n = bucket.previous_allocated_count;
if (n == 0)
{
n = m_options.min_blocks_per_chunk;
if (n < (m_options.min_bytes_per_chunk >> bytes_log2))
{
n = m_options.min_bytes_per_chunk >> bytes_log2;
}
}
else
{
n = n * 3 / 2;
if (n > (m_options.max_bytes_per_chunk >> bytes_log2))
{
n = m_options.max_bytes_per_chunk >> bytes_log2;
}
if (n > m_options.max_blocks_per_chunk)
{
n = m_options.max_blocks_per_chunk;
}
}

bytes = n << bytes_log2;

assert(n >= m_options.min_blocks_per_chunk);
assert(n <= m_options.max_blocks_per_chunk);
assert(bytes >= m_options.min_bytes_per_chunk);
assert(bytes <= m_options.max_bytes_per_chunk);

chunk_descriptor allocated;
allocated.size = bytes;
allocated.pointer = m_upstream->do_allocate(bytes, m_options.alignment);
m_allocated.push_back(allocated);
bucket.previous_allocated_count = n;

for (std::size_t i = 0; i < n; ++i)
{
bucket.free_blocks.push_back(
static_cast<void_ptr>(
static_cast<char_ptr>(allocated.pointer) + i * bucket_size
)
);
}
}

void_ptr ret = bucket.free_blocks.back();
bucket.free_blocks.pop_back();
return ret;
}

virtual void do_deallocate(void_ptr p, std::size_t n, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
n = (std::max)(n, m_options.smallest_block_size);
assert(detail::is_power_of_2(alignment));

assert(reinterpret_cast<detail::intmax_t>(detail::pointer_traits<void_ptr>::get(p)) % alignment == 0);

if (n > m_options.largest_block_size || alignment > m_options.alignment)
{
typename oversized_block_vector::iterator it = find_if(m_oversized.begin(), m_oversized.end(), equal_pointers(p));
assert(it != m_oversized.end());

oversized_block_descriptor oversized = *it;

if (m_options.cache_oversized)
{
typename oversized_block_vector::iterator position = lower_bound(m_cached_oversized.begin(), m_cached_oversized.end(), oversized);
m_cached_oversized.insert(position, oversized);
return;
}

m_oversized.erase(it);

m_upstream->do_deallocate(p, oversized.size, oversized.alignment);

return;
}

std::size_t n_log2 = hydra_thrust::detail::log2_ri(n);
std::size_t bucket_idx = n_log2 - m_smallest_block_log2;
pool & bucket = m_pools[bucket_idx];

bucket.free_blocks.push_back(p);
}
};



} 
} 

