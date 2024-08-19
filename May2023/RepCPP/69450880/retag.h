

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/tagged_iterator.h>
#include <hydra/detail/external/hydra_thrust/detail/pointer.h>

namespace hydra_thrust
{
namespace detail
{


template<typename FromTag, typename ToTag>
struct is_retaggable
: integral_constant<
bool,
(is_convertible<FromTag,ToTag>::value || is_convertible<ToTag,FromTag>::value)
>
{};


template<typename FromTag, typename ToTag, typename Result>
struct enable_if_retaggable
: enable_if<
is_retaggable<FromTag,ToTag>::value,
Result
>
{}; 


} 


template<typename Tag, typename Iterator>
__host__ __device__
hydra_thrust::detail::tagged_iterator<Iterator,Tag>
reinterpret_tag(Iterator iter)
{
return hydra_thrust::detail::tagged_iterator<Iterator,Tag>(iter);
} 


template<typename Tag, typename T>
__host__ __device__
hydra_thrust::pointer<T,Tag>
reinterpret_tag(T *ptr)
{
return hydra_thrust::pointer<T,Tag>(ptr);
} 


template<typename Tag, typename T, typename OtherTag, typename Reference, typename Derived>
__host__ __device__
hydra_thrust::pointer<T,Tag>
reinterpret_tag(hydra_thrust::pointer<T,OtherTag,Reference,Derived> ptr)
{
return reinterpret_tag<Tag>(ptr.get());
} 


template<typename Tag, typename BaseIterator, typename OtherTag>
__host__ __device__
hydra_thrust::detail::tagged_iterator<BaseIterator,Tag>
reinterpret_tag(hydra_thrust::detail::tagged_iterator<BaseIterator,OtherTag> iter)
{
return reinterpret_tag<Tag>(iter.base());
} 


template<typename Tag, typename Iterator>
__host__ __device__
typename hydra_thrust::detail::enable_if_retaggable<
typename hydra_thrust::iterator_system<Iterator>::type,
Tag,
hydra_thrust::detail::tagged_iterator<Iterator,Tag>
>::type
retag(Iterator iter)
{
return reinterpret_tag<Tag>(iter);
} 


template<typename Tag, typename T>
__host__ __device__
typename hydra_thrust::detail::enable_if_retaggable<
typename hydra_thrust::iterator_system<T*>::type,
Tag,
hydra_thrust::pointer<T,Tag>
>::type
retag(T *ptr)
{
return reinterpret_tag<Tag>(ptr);
} 


template<typename Tag, typename T, typename OtherTag>
__host__ __device__
typename hydra_thrust::detail::enable_if_retaggable<
OtherTag,
Tag,
hydra_thrust::pointer<T,Tag>
>::type
retag(hydra_thrust::pointer<T,OtherTag> ptr)
{
return reinterpret_tag<Tag>(ptr);
} 


template<typename Tag, typename BaseIterator, typename OtherTag>
__host__ __device__
typename hydra_thrust::detail::enable_if_retaggable<
OtherTag,
Tag,
hydra_thrust::detail::tagged_iterator<BaseIterator,Tag>
>::type
retag(hydra_thrust::detail::tagged_iterator<BaseIterator,OtherTag> iter)
{
return reinterpret_tag<Tag>(iter);
} 


} 

