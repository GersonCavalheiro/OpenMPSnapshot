
#ifndef BOOST_TT_DETAIL_ICE_EQ_HPP_INCLUDED
#define BOOST_TT_DETAIL_ICE_EQ_HPP_INCLUDED

#include <boost/config.hpp>

#if defined(__GNUC__) || defined(_MSC_VER)
# pragma message("NOTE: Use of this header (ice_eq.hpp) is deprecated")
#endif

namespace boost {
namespace type_traits {

template <int b1, int b2>
struct ice_eq
{
BOOST_STATIC_CONSTANT(bool, value = (b1 == b2));
};

template <int b1, int b2>
struct ice_ne
{
BOOST_STATIC_CONSTANT(bool, value = (b1 != b2));
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <int b1, int b2> bool const ice_eq<b1,b2>::value;
template <int b1, int b2> bool const ice_ne<b1,b2>::value;
#endif

} 
} 

#endif 
