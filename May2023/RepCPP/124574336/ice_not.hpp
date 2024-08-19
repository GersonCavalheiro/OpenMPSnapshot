
#ifndef BOOST_TT_DETAIL_ICE_NOT_HPP_INCLUDED
#define BOOST_TT_DETAIL_ICE_NOT_HPP_INCLUDED

#include <boost/config.hpp>

#if defined(__GNUC__) || defined(_MSC_VER)
# pragma message("NOTE: Use of this header (ice_not.hpp) is deprecated")
#endif

namespace boost {
namespace type_traits {

template <bool b>
struct ice_not
{
BOOST_STATIC_CONSTANT(bool, value = true);
};

template <>
struct ice_not<true>
{
BOOST_STATIC_CONSTANT(bool, value = false);
};

} 
} 

#endif 
