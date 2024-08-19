
#ifndef BOOST_INTRUSIVE_DETAIL_EQUAL_TO_VALUE_HPP
#define BOOST_INTRUSIVE_DETAIL_EQUAL_TO_VALUE_HPP

#ifndef BOOST_CONFIG_HPP
#  include <boost/config.hpp>
#endif

#if defined(BOOST_HAS_PRAGMA_ONCE)
#  pragma once
#endif

#include <boost/intrusive/detail/workaround.hpp>

namespace boost {
namespace intrusive {
namespace detail {

template<class ConstReference>
class equal_to_value
{
ConstReference t_;

public:
equal_to_value(ConstReference t)
:  t_(t)
{}

BOOST_INTRUSIVE_FORCEINLINE bool operator()(ConstReference t)const
{  return t_ == t;   }
};

}  
}  
}  

#endif 
