

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace detail
{

template<typename BaseAllocator>
struct no_throw_allocator : BaseAllocator
{
private:
typedef BaseAllocator super_t;

public:
inline __host__ __device__
no_throw_allocator(const BaseAllocator &other = BaseAllocator())
: super_t(other)
{}

template<typename U>
struct rebind
{
typedef no_throw_allocator<typename super_t::template rebind<U>::other> other;
}; 

__host__ __device__
void deallocate(typename super_t::pointer p, typename super_t::size_type n)
{
#ifndef __CUDA_ARCH__
try
{
super_t::deallocate(p, n);
} 
catch(...)
{
} 
#else
super_t::deallocate(p, n);
#endif
} 

inline __host__ __device__
bool operator==(no_throw_allocator const &other) { return super_t::operator==(other); }

inline __host__ __device__
bool operator!=(no_throw_allocator const &other) { return super_t::operator!=(other); }
}; 

} 
} 


