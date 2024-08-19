


#ifndef BOOST_ATOMIC_DETAIL_TYPE_TRAITS_IS_FLOATING_POINT_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_TYPE_TRAITS_IS_FLOATING_POINT_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_BASIC_HDR_TYPE_TRAITS) && !defined(BOOST_HAS_FLOAT128)
#include <type_traits>
#else
#include <boost/type_traits/is_floating_point.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_BASIC_HDR_TYPE_TRAITS) && !defined(BOOST_HAS_FLOAT128)
using std::is_floating_point;
#else
using boost::is_floating_point;
#endif

} 
} 
} 

#endif 
