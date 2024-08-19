






#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/type_traits/void_t.h>

#include <iterator>

namespace hydra_thrust
{

namespace detail
{

template <typename T, typename = void>
struct iterator_traits_impl {};

template <typename T>
struct iterator_traits_impl<
T
, typename voider<
typename T::difference_type
, typename T::value_type
, typename T::pointer
, typename T::reference
, typename T::iterator_category
>::type 
>
{
typedef typename T::difference_type difference_type;
typedef typename T::value_type value_type;
typedef typename T::pointer pointer;
typedef typename T::reference reference;
typedef typename T::iterator_category iterator_category;
};

} 


template <typename T>
struct iterator_traits : detail::iterator_traits_impl<T> {};

template<typename T>
struct iterator_traits<T*>
{
typedef std::ptrdiff_t difference_type;
typedef T value_type;
typedef T* pointer;
typedef T& reference;
typedef std::random_access_iterator_tag iterator_category;
};

template<typename T>
struct iterator_traits<const T*>
{
typedef std::ptrdiff_t difference_type;
typedef T value_type;
typedef const T* pointer;
typedef const T& reference;
typedef std::random_access_iterator_tag iterator_category;
}; 

template<typename Iterator> struct iterator_value;

template<typename Iterator> struct iterator_pointer;

template<typename Iterator> struct iterator_reference;

template<typename Iterator> struct iterator_difference;

template<typename Iterator> struct iterator_traversal;

template<typename Iterator> struct iterator_system;

} 

#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traversal_tags.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/host_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/device_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_system_tag.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/iterator_traits.inl>

