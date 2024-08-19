
#ifndef BOOST_ASIO_ASSOCIATED_ALLOCATOR_HPP
#define BOOST_ASIO_ASSOCIATED_ALLOCATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <memory>
#include <boost/asio/detail/type_traits.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T, typename E, typename = void>
struct associated_allocator_impl
{
typedef E type;

static type get(const T&, const E& e) BOOST_ASIO_NOEXCEPT
{
return e;
}
};

template <typename T, typename E>
struct associated_allocator_impl<T, E,
typename void_type<typename T::allocator_type>::type>
{
typedef typename T::allocator_type type;

static type get(const T& t, const E&) BOOST_ASIO_NOEXCEPT
{
return t.get_allocator();
}
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
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return detail::associated_allocator_impl<T, Allocator>::get(t, a);
}
};


template <typename T>
inline typename associated_allocator<T>::type
get_associated_allocator(const T& t) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<T>::get(t);
}


template <typename T, typename Allocator>
inline typename associated_allocator<T, Allocator>::type
get_associated_allocator(const T& t, const Allocator& a) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<T, Allocator>::get(t, a);
}

#if defined(BOOST_ASIO_HAS_ALIAS_TEMPLATES)

template <typename T, typename Allocator = std::allocator<void> >
using associated_allocator_t
= typename associated_allocator<T, Allocator>::type;

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
