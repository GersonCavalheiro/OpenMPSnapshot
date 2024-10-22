
#ifndef BOOST_TT_DETAIL_ICE_AND_HPP_INCLUDED
#define BOOST_TT_DETAIL_ICE_AND_HPP_INCLUDED

#include <boost/config.hpp>

#if defined(__GNUC__) || defined(_MSC_VER)
# pragma message("NOTE: Use of this header (ice_and.hpp) is deprecated")
#endif

namespace boost {
namespace type_traits {

template <bool b1, bool b2, bool b3 = true, bool b4 = true, bool b5 = true, bool b6 = true, bool b7 = true>
struct ice_and;

template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
struct ice_and
{
BOOST_STATIC_CONSTANT(bool, value = false);
};

template <>
struct ice_and<true, true, true, true, true, true, true>
{
BOOST_STATIC_CONSTANT(bool, value = true);
};

} 
} 

#endif 
