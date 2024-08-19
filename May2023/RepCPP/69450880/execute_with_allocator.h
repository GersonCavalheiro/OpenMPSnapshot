

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <hydra/detail/external/hydra_thrust/detail/execute_with_allocator_fwd.h>
#include <hydra/detail/external/hydra_thrust/pair.h>
#include <hydra/detail/external/hydra_thrust/detail/raw_pointer_cast.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator/allocator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/integer_math.h>

namespace hydra_thrust
{
namespace detail
{

template <
typename T
, typename Allocator
, template <typename> class BaseSystem
>
__host__
hydra_thrust::pair<T*, std::ptrdiff_t>
get_temporary_buffer(
hydra_thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system
, std::ptrdiff_t n
)
{
typedef typename hydra_thrust::detail::remove_reference<Allocator>::type naked_allocator;
typedef typename hydra_thrust::detail::allocator_traits<naked_allocator> alloc_traits;
typedef typename alloc_traits::void_pointer                        void_pointer;
typedef typename alloc_traits::size_type                           size_type;
typedef typename alloc_traits::value_type                          value_type;

size_type num_elements = divide_ri(sizeof(T) * n, sizeof(value_type));

void_pointer ptr = alloc_traits::allocate(system.get_allocator(), num_elements);

return hydra_thrust::make_pair(hydra_thrust::reinterpret_pointer_cast<T*>(ptr),n);
}

template <
typename Pointer
, typename Allocator
, template <typename> class BaseSystem
>
__host__
void
return_temporary_buffer(
hydra_thrust::detail::execute_with_allocator<Allocator, BaseSystem>& system
, Pointer p
)
{
typedef typename hydra_thrust::detail::remove_reference<Allocator>::type naked_allocator;
typedef typename hydra_thrust::detail::allocator_traits<naked_allocator> alloc_traits;
typedef typename alloc_traits::pointer                             pointer;

pointer to_ptr = hydra_thrust::reinterpret_pointer_cast<pointer>(p);
alloc_traits::deallocate(system.get_allocator(), to_ptr, 0);
}

#if __cplusplus >= 201103L

template <
typename T,
template <typename> class BaseSystem,
typename Allocator,
typename ...Dependencies
>
__host__
hydra_thrust::pair<T*, std::ptrdiff_t>
get_temporary_buffer(
hydra_thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>& system,
std::ptrdiff_t n
)
{
typedef typename hydra_thrust::detail::remove_reference<Allocator>::type naked_allocator;
typedef typename hydra_thrust::detail::allocator_traits<naked_allocator> alloc_traits;
typedef typename alloc_traits::void_pointer                        void_pointer;
typedef typename alloc_traits::size_type                           size_type;
typedef typename alloc_traits::value_type                          value_type;

size_type num_elements = divide_ri(sizeof(T) * n, sizeof(value_type));

void_pointer ptr = alloc_traits::allocate(system.get_allocator(), num_elements);

return hydra_thrust::make_pair(hydra_thrust::reinterpret_pointer_cast<T*>(ptr),n);
}

template <
typename Pointer,
template <typename> class BaseSystem,
typename Allocator,
typename ...Dependencies
>
__host__
void
return_temporary_buffer(
hydra_thrust::detail::execute_with_allocator_and_dependencies<Allocator, BaseSystem, Dependencies...>& system,
Pointer p
)
{
typedef typename hydra_thrust::detail::remove_reference<Allocator>::type naked_allocator;
typedef typename hydra_thrust::detail::allocator_traits<naked_allocator> alloc_traits;
typedef typename alloc_traits::pointer                             pointer;

pointer to_ptr = hydra_thrust::reinterpret_pointer_cast<pointer>(p);
alloc_traits::deallocate(system.get_allocator(), to_ptr, 0);
}

#endif

}} 

