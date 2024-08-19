

#if !defined(BOOST_DETAIL_CONTAINER_FWD_HPP)
#define BOOST_DETAIL_CONTAINER_FWD_HPP

#if defined(_MSC_VER) && \
!defined(BOOST_DETAIL_TEST_CONFIG_ONLY)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>


#if !defined(BOOST_DETAIL_NO_CONTAINER_FWD)
#  if defined(BOOST_DETAIL_CONTAINER_FWD)
#  elif defined(__SGI_STL_PORT) || defined(_STLPORT_VERSION)
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  elif defined(__LIBCOMO__)
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  elif defined(__STD_RWCOMPILER_H__) || defined(_RWSTD_VER)
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  elif defined(_LIBCPP_VERSION)
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  elif defined(__GLIBCPP__) || defined(__GLIBCXX__)
#    if __GLIBCXX__ >= 20070513 \
|| defined(_GLIBCXX_DEBUG) \
|| defined(_GLIBCXX_PARALLEL) \
|| defined(_GLIBCXX_PROFILE)
#      define BOOST_DETAIL_NO_CONTAINER_FWD
#    else
#      if defined(__GLIBCXX__) && __GLIBCXX__ >= 20040530
#        define BOOST_CONTAINER_FWD_COMPLEX_STRUCT
#      endif
#    endif
#  elif defined(__STL_CONFIG_H)
#    define BOOST_CONTAINER_FWD_BAD_BITSET
#    if !defined(__STL_NON_TYPE_TMPL_PARAM_BUG)
#      define BOOST_CONTAINER_FWD_BAD_DEQUE
#     endif
#  elif defined(__MSL_CPP__)
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  elif defined(__IBMCPP__)
#  elif defined(MSIPL_COMPILE_H)
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  elif (defined(_YVALS) && !defined(__IBMCPP__)) || defined(_CPPLIB_VER)
#  else
#    define BOOST_DETAIL_NO_CONTAINER_FWD
#  endif
#endif

#if !defined(BOOST_DETAIL_TEST_CONFIG_ONLY)

#if defined(BOOST_DETAIL_NO_CONTAINER_FWD) && \
!defined(BOOST_DETAIL_TEST_FORCE_CONTAINER_FWD)

#include <deque>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <bitset>
#include <string>
#include <complex>

#else

#include <cstddef>

#if defined(BOOST_CONTAINER_FWD_BAD_DEQUE)
#include <deque>
#endif

#if defined(BOOST_CONTAINER_FWD_BAD_BITSET)
#include <bitset>
#endif

#if defined(BOOST_MSVC)
#pragma warning(push)
#pragma warning(disable:4099) 
#endif

namespace std
{
template <class T> class allocator;
template <class charT, class traits, class Allocator> class basic_string;

template <class charT> struct char_traits;

#if defined(BOOST_CONTAINER_FWD_COMPLEX_STRUCT)
template <class T> struct complex;
#else
template <class T> class complex;
#endif

#if !defined(BOOST_CONTAINER_FWD_BAD_DEQUE)
template <class T, class Allocator> class deque;
#endif

template <class T, class Allocator> class list;
template <class T, class Allocator> class vector;
template <class Key, class T, class Compare, class Allocator> class map;
template <class Key, class T, class Compare, class Allocator>
class multimap;
template <class Key, class Compare, class Allocator> class set;
template <class Key, class Compare, class Allocator> class multiset;

#if !defined(BOOST_CONTAINER_FWD_BAD_BITSET)
template <size_t N> class bitset;
#endif
template <class T1, class T2> struct pair;
}

#if defined(BOOST_MSVC)
#pragma warning(pop)
#endif

#endif 

#endif 

#endif
