#ifndef BOOST_CONTAINER_DETAIL_VALUE_FUNCTORS_HPP
#define BOOST_CONTAINER_DETAIL_VALUE_FUNCTORS_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

namespace boost {
namespace container {

template<class ValueType>
struct value_less
{
bool operator()(const ValueType &a, const ValueType &b) const
{  return a < b;  }
};

template<class ValueType>
struct value_equal
{
bool operator()(const ValueType &a, const ValueType &b) const
{  return a == b;  }
};

}  
}  

#endif   
