
#ifndef BOOST_INTERPROCESS_DETAIL_MIN_MAX_HPP
#define BOOST_INTERPROCESS_DETAIL_MIN_MAX_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif
#
#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/interprocess/detail/config_begin.hpp>
#include <boost/interprocess/detail/workaround.hpp>

namespace boost {
namespace interprocess {

template<class T>
const T &max_value(const T &a, const T &b)
{  return a > b ? a : b;   }

template<class T>
const T &min_value(const T &a, const T &b)
{  return a < b ? a : b;   }

}  
}  

#include <boost/interprocess/detail/config_end.hpp>

#endif   

