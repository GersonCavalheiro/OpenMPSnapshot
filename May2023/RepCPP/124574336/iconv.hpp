
#ifndef BOOST_MATH_ICONV_HPP
#define BOOST_MATH_ICONV_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/round.hpp>
#include <boost/type_traits/is_convertible.hpp>

namespace boost { namespace math { namespace detail{

template <class T, class Policy>
inline int iconv_imp(T v, Policy const&, boost::true_type const&)
{
return static_cast<int>(v);
}

template <class T, class Policy>
inline int iconv_imp(T v, Policy const& pol, boost::false_type const&)
{
BOOST_MATH_STD_USING
return iround(v, pol);
}

template <class T, class Policy>
inline int iconv(T v, Policy const& pol)
{
typedef typename boost::is_convertible<T, int>::type tag_type;
return iconv_imp(v, pol, tag_type());
}


}}} 

#endif 

