


#ifndef BOOST_ATOMIC_DETAIL_TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_TYPE_TRAITS_IS_TRIVIALLY_COPYABLE_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#if !defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS)
#include <type_traits>
#else
#define BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

#if !defined(BOOST_ATOMIC_DETAIL_NO_CXX11_IS_TRIVIALLY_COPYABLE)

namespace boost {
namespace atomics {
namespace detail {

using std::is_trivially_copyable;

} 
} 
} 

#endif 

#endif 
