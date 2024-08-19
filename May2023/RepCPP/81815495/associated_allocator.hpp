
#ifndef ASIO_ASSOCIATED_ALLOCATOR_HPP
#define ASIO_ASSOCIATED_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <memory>
#include "asio/associator.hpp"
#include "asio/detail/functional.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

template <typename T, typename Allocator>
struct associated_allocator;

namespace detail {

template <typename T, typename = void>
struct has_allocator_type : false_type
{
};

template <typename T>
struct has_allocator_type<T,
typename void_type<typename T::executor_type>::type>
: true_type
{
};

template <typename T, typename E, typename = void, typename = void>
struct associated_allocator_impl
{
typedef E type;

static type get(const T&, const E& e) ASIO_NOEXCEPT
{
return e;
}
};

template <typename T, typename E>
struct associated_allocator_impl<T, E,
typename void_type<typename T::allocator_type>::type>
{
typedef typename T::allocator_type type;

static type get(const T& t, const E&) ASIO_NOEXCEPT
{
return t.get_allocator();
}
};

template <typename T, typename E>
struct associated_allocator_impl<T, E,
typename enable_if<
!has_allocator_type<T>::value
>::type,
typename void_type<
typename associator<associated_allocator, T, E>::type
>::type> : associator<associated_allocator, T, E>
{
};

} 


template <typename T, typename Allocator = std::allocator<void> >
struct associated_allocator
{
#if defined(GENERATING_DOCUMENTATION)
typedef see_below type;
#else 
typedef typename detail::associated_allocator_impl<T, Allocator>::type type;
#endif 

static type get(const T& t,
const Allocator& a = Allocator()) ASIO_NOEXCEPT
{
return detail::associated_allocator_impl<T, Allocator>::get(t, a);
}
};


template <typename T>
inline typename associated_allocator<T>::type
get_associated_allocator(const T& t) ASIO_NOEXCEPT
{
return associated_allocator<T>::get(t);
}


template <typename T, typename Allocator>
inline typename associated_allocator<T, Allocator>::type
get_associated_allocator(const T& t, const Allocator& a) ASIO_NOEXCEPT
{
return associated_allocator<T, Allocator>::get(t, a);
}

#if defined(ASIO_HAS_ALIAS_TEMPLATES)

template <typename T, typename Allocator = std::allocator<void> >
using associated_allocator_t
= typename associated_allocator<T, Allocator>::type;

#endif 

#if defined(ASIO_HAS_STD_REFERENCE_WRAPPER) \
|| defined(GENERATING_DOCUMENTATION)

template <typename T, typename Allocator>
struct associated_allocator<reference_wrapper<T>, Allocator>
{
typedef typename associated_allocator<T, Allocator>::type type;

static type get(reference_wrapper<T> t,
const Allocator& a = Allocator()) ASIO_NOEXCEPT
{
return associated_allocator<T, Allocator>::get(t.get(), a);
}
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
