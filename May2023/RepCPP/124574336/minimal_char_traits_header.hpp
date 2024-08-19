#ifndef BOOST_CONTAINER_DETAIL_MINIMAL_CHAR_TRAITS_HEADER_HPP
#define BOOST_CONTAINER_DETAIL_MINIMAL_CHAR_TRAITS_HEADER_HPP
#
#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif
#
#
#if defined(_MSC_VER) && defined(BOOST_DINKUMWARE_STDLIB)
#include <iosfwd>   
#elif defined(BOOST_GNU_STDLIB)
#include <bits/char_traits.h>
#else
#include <string>  
#endif

#endif 
