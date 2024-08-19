#ifndef BOOST_INTERPROCESS_DETAIL_SWAP_HPP
#define BOOST_INTERPROCESS_DETAIL_SWAP_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost { namespace interprocess {

template<class T>
void simple_swap(T&x, T&y)
{  T tmp(x); x = y; y = tmp;  }

}}  

#endif 
