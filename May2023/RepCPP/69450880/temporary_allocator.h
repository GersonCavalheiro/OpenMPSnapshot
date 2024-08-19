

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/tagged_allocator.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/allocator_traits.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/memory.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>

namespace hydra_thrust
{
namespace detail
{


template<typename T, typename System>
class temporary_allocator
: public hydra_thrust::detail::tagged_allocator<
T, System, hydra_thrust::pointer<T,System>
>
{
private:
typedef hydra_thrust::detail::tagged_allocator<
T, System, hydra_thrust::pointer<T,System>
> super_t;

System &m_system;

public:
typedef typename super_t::pointer   pointer;
typedef typename super_t::size_type size_type;

inline __host__ __device__
temporary_allocator(const temporary_allocator &other) :
super_t(),
m_system(other.m_system)
{}

inline __host__ __device__
explicit temporary_allocator(hydra_thrust::execution_policy<System> &system) :
super_t(),
m_system(hydra_thrust::detail::derived_cast(system))
{}

__host__ __device__
pointer allocate(size_type cnt);

__host__ __device__
void deallocate(pointer p, size_type n);

__host__ __device__
inline System &system()
{
return m_system;
} 

private:
typedef hydra_thrust::pair<pointer, size_type> pointer_and_size;
}; 


} 
} 

#include <hydra/detail/external/hydra_thrust/detail/allocator/temporary_allocator.inl>

