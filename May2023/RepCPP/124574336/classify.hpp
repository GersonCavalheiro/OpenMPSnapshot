


#ifndef BOOST_ATOMIC_DETAIL_CLASSIFY_HPP_INCLUDED_
#define BOOST_ATOMIC_DETAIL_CLASSIFY_HPP_INCLUDED_

#include <boost/atomic/detail/config.hpp>
#include <boost/atomic/detail/type_traits/is_integral.hpp>
#include <boost/atomic/detail/type_traits/is_function.hpp>
#include <boost/atomic/detail/type_traits/is_floating_point.hpp>
#include <boost/atomic/detail/header.hpp>

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {
namespace atomics {
namespace detail {

template< typename T, bool IsFunction = atomics::detail::is_function< T >::value >
struct classify_pointer
{
typedef void* type;
};

template< typename T >
struct classify_pointer< T, true >
{
typedef void type;
};

template< typename T, bool IsInt = atomics::detail::is_integral< T >::value, bool IsFloat = atomics::detail::is_floating_point< T >::value >
struct classify
{
typedef void type;
};

template< typename T >
struct classify< T, true, false > { typedef int type; };

#if !defined(BOOST_ATOMIC_NO_FLOATING_POINT)
template< typename T >
struct classify< T, false, true > { typedef float type; };
#endif

template< typename T >
struct classify< T*, false, false > { typedef typename classify_pointer< T >::type type; };

template< >
struct classify< void*, false, false > { typedef void type; };

template< >
struct classify< const void*, false, false > { typedef void type; };

template< >
struct classify< volatile void*, false, false > { typedef void type; };

template< >
struct classify< const volatile void*, false, false > { typedef void type; };

template< typename T, typename U >
struct classify< T U::*, false, false > { typedef void type; };

} 
} 
} 

#include <boost/atomic/detail/footer.hpp>

#endif 
