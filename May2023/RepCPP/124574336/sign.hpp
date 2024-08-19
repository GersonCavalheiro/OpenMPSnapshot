
#ifndef BOOST_SPIRIT_SUPPORT_DETAIL_SIGN_HPP
#define BOOST_SPIRIT_SUPPORT_DETAIL_SIGN_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/math/special_functions/sign.hpp>

namespace boost { namespace spirit { namespace detail
{
template<typename T> 
inline bool (signbit)(T x)
{
return (boost::math::signbit)(x) ? true : false;
}

template<typename T> 
inline T (changesign)(T x)
{
return (boost::math::changesign)(x);
}

}}}

#endif
