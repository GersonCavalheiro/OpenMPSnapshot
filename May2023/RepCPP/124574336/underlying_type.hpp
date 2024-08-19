


#ifndef BOOST_CORE_UNDERLYING_TYPE_HPP
#define BOOST_CORE_UNDERLYING_TYPE_HPP

#include <boost/config.hpp>

#if !defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS) || (defined(BOOST_GCC) && BOOST_GCC >= 40700 && defined(__GXX_EXPERIMENTAL_CXX0X__))
#include <type_traits>
#define BOOST_DETAIL_HAS_STD_UNDERLYING_TYPE
#endif

#ifdef BOOST_HAS_PRAGMA_ONCE
#pragma once
#endif

namespace boost {

namespace detail {

template< typename EnumType, typename Void = void >
struct underlying_type_impl;

#if defined(BOOST_NO_CXX11_SCOPED_ENUMS)

template< typename EnumType >
struct underlying_type_impl< EnumType, typename EnumType::is_boost_scoped_enum_tag >
{

typedef typename EnumType::underlying_type type;
};

#endif

#if defined(BOOST_DETAIL_HAS_STD_UNDERLYING_TYPE)

template< typename EnumType, typename Void >
struct underlying_type_impl
{
typedef typename std::underlying_type< EnumType >::type type;
};

#endif

} 

#if !defined(BOOST_NO_CXX11_SCOPED_ENUMS) && !defined(BOOST_DETAIL_HAS_STD_UNDERLYING_TYPE)
#define BOOST_NO_UNDERLYING_TYPE
#endif


template< typename EnumType >
struct underlying_type :
public detail::underlying_type_impl< EnumType >
{
};

} 

#endif  
