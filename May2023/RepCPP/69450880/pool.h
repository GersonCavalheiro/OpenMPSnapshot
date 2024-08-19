



#pragma once

#include <algorithm>

#include <hydra/detail/external/hydra_thrust/host_vector.h>

#include <hydra/detail/external/hydra_thrust/mr/memory_resource.h>
#include <hydra/detail/external/hydra_thrust/mr/allocator.h>
#include <hydra/detail/external/hydra_thrust/mr/pool_options.h>

#include <cassert>

namespace hydra_thrust
{
namespace mr
{




template<typename Upstream>
class unsynchronized_pool_resource HYDRA_THRUST_FINAL
: public memory_resource<typename Upstream::pointer>,
private validator<Upstream>
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


unsynchronized_pool_resource(Upstream * upstream, pool_options options = get_default_options())
: m_upstream(upstream),
m_options(options),
m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
m_pools(upstream),
m_allocated(),
m_oversized(),
m_cached_oversized()
{
assert(m_options.validate());

pool p = { block_descriptor_ptr(), 0 };
m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
}



unsynchronized_pool_resource(pool_options options = get_default_options())
: m_upstream(get_global_resource<Upstream>()),
m_options(options),
m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
m_pools(get_global_resource<Upstream>()),
m_allocated(),
m_oversized(),
m_cached_oversized()
{
assert(m_options.validate());

pool p = { block_descriptor_ptr(), 0 };
m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
}


~unsynchronized_pool_resource()
{
release();
}

private:
typedef typename Upstream::pointer void_ptr;
typedef typename hydra_thrust::detail::pointer_traits<void_ptr>::template rebind<char>::other char_ptr;

struct block_descriptor;
struct chunk_descriptor;
struct oversized_block_descriptor;

typedef typename hydra_thrust::detail::pointer_traits<void_ptr>::template rebind<block_descriptor>::other block_descriptor_ptr;
typedef typename hydra_thrust::detail::pointer_traits<void_ptr>::template rebind<chunk_descriptor>::other chunk_descriptor_ptr;
typedef typename hydra_thrust::detail::pointer_traits<void_ptr>::template rebind<oversized_block_descriptor>::other oversized_block_descriptor_ptr;

struct block_descriptor
{
block_descriptor_ptr next;
};

struct chunk_descriptor
{
std::size_t size;
chunk_descriptor_ptr next;
};

struct oversized_block_descriptor
{
std::size_t size;
std::size_t alignment;
oversized_block_descriptor_ptr prev;
oversized_block_descriptor_ptr next;
oversized_block_descriptor_ptr next_cached;
};

struct pool
{
block_descriptor_ptr free_list;
std::size_t previous_allocated_count;
};

typedef hydra_thrust::host_vector<
pool,
allocator<pool, Upstream>
> pool_vector;

Upstream * m_upstream;

pool_options m_options;
std::size_t m_smallest_block_log2;

pool_vector m_pools;
chunk_descriptor_ptr m_allocated;
oversized_block_descriptor_ptr m_oversized;
oversized_block_descriptor_ptr m_cached_oversized;

public:

void release()
{
for (std::size_t i = 0; i < m_pools.size(); ++i)
{
hydra_thrust::raw_reference_cast(m_pools[i]).free_list = block_descriptor_ptr();
hydra_thrust::raw_reference_cast(m_pools[i]).previous_allocated_count = 0;
}

while (detail::pointer_traits<chunk_descriptor_ptr>::get(m_allocated))
{
chunk_descriptor_ptr alloc = m_allocated;
m_allocated = hydra_thrust::raw_reference_cast(*m_allocated).next;

void_ptr p = static_cast<void_ptr>(
static_cast<char_ptr>(
static_cast<void_ptr>(alloc)
) - hydra_thrust::raw_reference_cast(*alloc).size
);
m_upstream->do_deallocate(p, hydra_thrust::raw_reference_cast(*alloc).size + sizeof(chunk_descriptor), m_options.alignment);
}

while (detail::pointer_traits<oversized_block_descriptor_ptr>::get(m_oversized))
{
oversized_block_descriptor_ptr alloc = m_oversized;
m_oversized = hydra_thrust::raw_reference_cast(*m_oversized).next;

void_ptr p = static_cast<void_ptr>(
static_cast<char_ptr>(
static_cast<void_ptr>(alloc)
) - hydra_thrust::raw_reference_cast(*alloc).size
);
m_upstream->do_deallocate(p, hydra_thrust::raw_reference_cast(*alloc).size + sizeof(oversized_block_descriptor), hydra_thrust::raw_reference_cast(*alloc).alignment);
}

m_cached_oversized = oversized_block_descriptor_ptr();
}

HYDRA_THRUST_NODISCARD virtual void_ptr do_allocate(std::size_t bytes, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
bytes = (std::max)(bytes, m_options.smallest_block_size);
assert(detail::is_power_of_2(alignment));

if (bytes > m_options.largest_block_size || alignment > m_options.alignment)
{
if (m_options.cache_oversized)
{
oversized_block_descriptor_ptr ptr = m_cached_oversized;
oversized_block_descriptor_ptr * previous = &m_cached_oversized;
while (detail::pointer_traits<oversized_block_descriptor_ptr>::get(ptr))
{
oversized_block_descriptor desc = *ptr;
bool is_good = desc.size >= bytes && desc.alignment >= alignment;

if (is_good)
{
std::size_t size_factor = desc.size / bytes;
if (size_factor >= m_options.cached_size_cutoff_factor)
{
is_good = false;
}
}

if (is_good)
{
std::size_t alignment_factor = desc.alignment / alignment;
if (alignment_factor >= m_options.cached_alignment_cutoff_factor)
{
is_good = false;
}
}

if (is_good)
{
if (previous != &m_cached_oversized)
{
oversized_block_descriptor previous_desc = **previous;
previous_desc.next_cached = desc.next_cached;
**previous = previous_desc;
}
else
{
m_cached_oversized = desc.next_cached;
}

desc.next_cached = oversized_block_descriptor_ptr();
*ptr = desc;

return static_cast<void_ptr>(
static_cast<char_ptr>(
static_cast<void_ptr>(ptr)
) - desc.size
);
}

previous = &hydra_thrust::raw_reference_cast(*ptr).next_cached;
ptr = *previous;
}
}

void_ptr allocated = m_upstream->do_allocate(bytes + sizeof(oversized_block_descriptor), alignment);
oversized_block_descriptor_ptr block = static_cast<oversized_block_descriptor_ptr>(
static_cast<void_ptr>(
static_cast<char_ptr>(allocated) + bytes
)
);

oversized_block_descriptor desc;
desc.size = bytes;
desc.alignment = alignment;
desc.prev = oversized_block_descriptor_ptr();
desc.next = m_oversized;
desc.next_cached = oversized_block_descriptor_ptr();
*block = desc;
m_oversized = block;

if (detail::pointer_traits<oversized_block_descriptor_ptr>::get(desc.next))
{
oversized_block_descriptor next = *desc.next;
next.prev = block;
*desc.next = next;
}

return allocated;
}

std::size_t bytes_log2 = hydra_thrust::detail::log2_ri(bytes);
std::size_t bucket_idx = bytes_log2 - m_smallest_block_log2;
pool & bucket = hydra_thrust::raw_reference_cast(m_pools[bucket_idx]);

bytes = static_cast<std::size_t>(1) << bytes_log2;

if (!detail::pointer_traits<block_descriptor_ptr>::get(bucket.free_list))
{
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

std::size_t descriptor_size = (std::max)(sizeof(block_descriptor), m_options.alignment);
std::size_t block_size = bytes + descriptor_size;
block_size += m_options.alignment - block_size % m_options.alignment;
std::size_t chunk_size = block_size * n;

void_ptr allocated = m_upstream->do_allocate(chunk_size + sizeof(chunk_descriptor), m_options.alignment);
chunk_descriptor_ptr chunk = static_cast<chunk_descriptor_ptr>(
static_cast<void_ptr>(
static_cast<char_ptr>(allocated) + chunk_size
)
);

chunk_descriptor desc;
desc.size = chunk_size;
desc.next = m_allocated;
*chunk = desc;
m_allocated = chunk;

for (std::size_t i = 0; i < n; ++i)
{
block_descriptor_ptr block = static_cast<block_descriptor_ptr>(
static_cast<void_ptr>(
static_cast<char_ptr>(allocated) + block_size * i + bytes
)
);

block_descriptor desc;
desc.next = bucket.free_list;
*block = desc;
bucket.free_list = block;
}
}

block_descriptor_ptr block = bucket.free_list;
bucket.free_list = hydra_thrust::raw_reference_cast(*block).next;
return static_cast<void_ptr>(
static_cast<char_ptr>(
static_cast<void_ptr>(block)
) - bytes
);
}

virtual void do_deallocate(void_ptr p, std::size_t n, std::size_t alignment = HYDRA_THRUST_MR_DEFAULT_ALIGNMENT) HYDRA_THRUST_OVERRIDE
{
n = (std::max)(n, m_options.smallest_block_size);
assert(detail::is_power_of_2(alignment));

assert(reinterpret_cast<detail::intmax_t>(detail::pointer_traits<void_ptr>::get(p)) % alignment == 0);

if (n > m_options.largest_block_size || alignment > m_options.alignment)
{
oversized_block_descriptor_ptr block = static_cast<oversized_block_descriptor_ptr>(
static_cast<void_ptr>(
static_cast<char_ptr>(p) + n
)
);

oversized_block_descriptor desc = *block;

if (m_options.cache_oversized)
{
desc.next_cached = m_cached_oversized;
*block = desc;
m_cached_oversized = block;

return;
}

if (!detail::pointer_traits<oversized_block_descriptor_ptr>::get(desc.prev))
{
assert(m_oversized == block);
m_oversized = desc.next;
}
else
{
oversized_block_descriptor prev = *desc.prev;
assert(prev.next == block);
prev.next = desc.next;
*desc.prev = prev;
}

if (detail::pointer_traits<oversized_block_descriptor_ptr>::get(desc.next))
{
oversized_block_descriptor next = *desc.next;
assert(next.prev == block);
next.prev = desc.prev;
*desc.next = next;
}

m_upstream->do_deallocate(p, desc.size + sizeof(oversized_block_descriptor), desc.alignment);

return;
}

std::size_t n_log2 = hydra_thrust::detail::log2_ri(n);
std::size_t bucket_idx = n_log2 - m_smallest_block_log2;
pool & bucket = hydra_thrust::raw_reference_cast(m_pools[bucket_idx]);

n = static_cast<std::size_t>(1) << n_log2;

block_descriptor_ptr block = static_cast<block_descriptor_ptr>(
static_cast<void_ptr>(
static_cast<char_ptr>(p) + n
)
);

block_descriptor desc;
desc.next = bucket.free_list;
*block = desc;
bucket.free_list = block;
}
};



} 
} 

