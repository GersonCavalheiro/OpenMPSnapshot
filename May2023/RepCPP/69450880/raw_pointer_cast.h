

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>

namespace hydra_thrust
{

template<typename Pointer>
__host__ __device__
typename hydra_thrust::detail::pointer_traits<Pointer>::raw_pointer
raw_pointer_cast(Pointer ptr)
{
return hydra_thrust::detail::pointer_traits<Pointer>::get(ptr);
}

template <typename ToPointer, typename FromPointer>
__host__ __device__
ToPointer
reinterpret_pointer_cast(FromPointer ptr)
{
typedef typename hydra_thrust::detail::pointer_element<ToPointer>::type to_element;
return ToPointer(reinterpret_cast<to_element*>(hydra_thrust::raw_pointer_cast(ptr)));
}

template <typename ToPointer, typename FromPointer>
__host__ __device__
ToPointer
static_pointer_cast(FromPointer ptr)
{
typedef typename hydra_thrust::detail::pointer_element<ToPointer>::type to_element;
return ToPointer(static_cast<to_element*>(hydra_thrust::raw_pointer_cast(ptr)));
}

} 

