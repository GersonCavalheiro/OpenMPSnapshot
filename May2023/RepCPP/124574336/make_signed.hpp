


#ifndef BOOST_ATOMIC_DETAIL_TYPE_TRAITS_MAKE_SIGNED_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_TYPE_TRAITS_MAKE_SIGNED_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_BASIC_HDR_TYPE_TRAITS) && !defined(BOOST_HAS_INT128)
#include <type_traits>
#else
#include <boost/type_traits/make_signed.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_BASIC_HDR_TYPE_TRAITS) && !defined(BOOST_HAS_INT128)
using std::make_signed;
#else
using boost::make_signed;
#endif

} 
} 
} 

#endif 
