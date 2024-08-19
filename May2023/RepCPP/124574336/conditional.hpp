


#ifndef BOOST_ATOMIC_DETAIL_TYPE_TRAITS_CONDITIONAL_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_TYPE_TRAITS_CONDITIONAL_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_BASIC_HDR_TYPE_TRAITS)
#include <type_traits>
#else
#include <boost/type_traits/conditional.hpp>
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_BASIC_HDR_TYPE_TRAITS)
using std::conditional;
#else
using boost::conditional;
#endif

} 
} 
} 

#endif 
