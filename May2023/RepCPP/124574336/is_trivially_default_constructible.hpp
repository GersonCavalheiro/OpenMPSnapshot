


#ifndef BOOST_ATOMIC_DETAIL_TYPE_TRAITS_IS_TRIVIALLY_DEFAULT_CONSTRUCTIBLE_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_TYPE_TRAITS_IS_TRIVIALLY_DEFAULT_CONSTRUCTIBLE_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#if !defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS)
#include <type_traits>
#else
#include <boost/type_traits/has_trivial_constructor.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

#if !defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS)
using std::is_trivially_default_constructible;
#elif !defined(BOOST_NO_CXX11_TEMPLATE_ALIASES)
template< typename T >
using is_trivially_default_constructible = boost::has_trivial_constructor< T >;
#else
template< typename T >
struct is_trivially_default_constructible : public boost::has_trivial_constructor< T > {};
#endif

} 
} 
} 

#endif 
