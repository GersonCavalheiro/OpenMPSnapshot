
#ifndef BOOST_MATH_ROUND_HPP
#define BOOST_MATH_ROUND_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace boost{ namespace math{

namespace detail{

template <class T, class Policy>
inline typename tools::promote_args<T>::type round(const T& v, const Policy& pol, const boost::false_type&)
{
BOOST_MATH_STD_USING
typedef typename tools::promote_args<T>::type result_type;
if(!(boost::math::isfinite)(v))
return policies::raise_rounding_error("boost::math::round<%1%>(%1%)", 0, static_cast<result_type>(v), static_cast<result_type>(v), pol);
if (-0.5 < v && v < 0.5)
{
return static_cast<result_type>(0);
}
else if (v > 0)
{
result_type c(ceil(v));
return 0.5 < c - v ? c - 1 : c;
}
else
{
result_type f(floor(v));
return 0.5 < v - f ? f + 1 : f;
}
}
template <class T, class Policy>
inline typename tools::promote_args<T>::type round(const T& v, const Policy&, const boost::true_type&)
{
return v;
}

} 

template <class T, class Policy>
inline typename tools::promote_args<T>::type round(const T& v, const Policy& pol)
{
return detail::round(v, pol, boost::integral_constant<bool, detail::is_integer_for_rounding<T>::value>());
}
template <class T>
inline typename tools::promote_args<T>::type round(const T& v)
{
return round(v, policies::policy<>());
}
template <class T, class Policy>
inline int iround(const T& v, const Policy& pol)
{
BOOST_MATH_STD_USING
T r = boost::math::round(v, pol);
if((r > (std::numeric_limits<int>::max)()) || (r < (std::numeric_limits<int>::min)()))
return static_cast<int>(policies::raise_rounding_error("boost::math::iround<%1%>(%1%)", 0, v, 0, pol));
return static_cast<int>(r);
}
template <class T>
inline int iround(const T& v)
{
return iround(v, policies::policy<>());
}

template <class T, class Policy>
inline long lround(const T& v, const Policy& pol)
{
BOOST_MATH_STD_USING
T r = boost::math::round(v, pol);
if((r > (std::numeric_limits<long>::max)()) || (r < (std::numeric_limits<long>::min)()))
return static_cast<long int>(policies::raise_rounding_error("boost::math::lround<%1%>(%1%)", 0, v, 0L, pol));
return static_cast<long int>(r);
}
template <class T>
inline long lround(const T& v)
{
return lround(v, policies::policy<>());
}

#ifdef BOOST_HAS_LONG_LONG

template <class T, class Policy>
inline boost::long_long_type llround(const T& v, const Policy& pol)
{
BOOST_MATH_STD_USING
T r = boost::math::round(v, pol);
if((r > (std::numeric_limits<boost::long_long_type>::max)()) || (r < (std::numeric_limits<boost::long_long_type>::min)()))
return static_cast<boost::long_long_type>(policies::raise_rounding_error("boost::math::llround<%1%>(%1%)", 0, v, static_cast<boost::long_long_type>(0), pol));
return static_cast<boost::long_long_type>(r);
}
template <class T>
inline boost::long_long_type llround(const T& v)
{
return llround(v, policies::policy<>());
}

#endif

}} 

#endif 
