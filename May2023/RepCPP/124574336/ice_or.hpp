
#ifndef BOOST_TT_DETAIL_ICE_OR_HPP_INCLUDED
#define BOOST_TT_DETAIL_ICE_OR_HPP_INCLUDED

#include <boost/config.hpp>

#if defined(__GNUC__) || defined(_MSC_VER)
# pragma message("NOTE: Use of this header (ice_or.hpp) is deprecated")
#endif

namespace boost {
namespace type_traits {

template <bool b1, bool b2, bool b3 = false, bool b4 = false, bool b5 = false, bool b6 = false, bool b7 = false>
struct ice_or;

template <bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7>
struct ice_or
{
BOOST_STATIC_CONSTANT(bool, value = true);
};

template <>
struct ice_or<false, false, false, false, false, false, false>
{
BOOST_STATIC_CONSTANT(bool, value = false);
};

} 
} 

#endif 
