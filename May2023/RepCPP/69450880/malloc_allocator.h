

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/tagged_allocator.h>

namespace hydra_thrust
{
namespace detail
{

template<typename T, typename System, typename Pointer>
class malloc_allocator
: public hydra_thrust::detail::tagged_allocator<
T, System, Pointer
>
{
private:
typedef hydra_thrust::detail::tagged_allocator<
T, System, Pointer
> super_t;

public:
typedef typename super_t::pointer   pointer;
typedef typename super_t::size_type size_type;

pointer allocate(size_type cnt);

void deallocate(pointer p, size_type n);
};

} 
} 

#include <hydra/detail/external/hydra_thrust/detail/allocator/malloc_allocator.inl>

