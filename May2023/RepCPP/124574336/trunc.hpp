
#ifndef BOOST_MATH_TRUNC_HPP
#define BOOST_MATH_TRUNC_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/type_traits/is_constructible.hpp>
#include <boost/core/enable_if.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
inline typename tools::promote_args<T>::type trunc(const T& v, const Policy& pol, const boost::false_type&)
{
BOOST_MATH_STD_USING
typedef typename tools::promote_args<T>::type result_type;
if(!(boost::math::isfinite)(v))
return policies::raise_rounding_error("boost::math::trunc<%1%>(%1%)", 0, static_cast<result_type>(v), static_cast<result_type>(v), pol);
return (v >= 0) ? static_cast<result_type>(floor(v)) : static_cast<result_type>(ceil(v));
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type trunc(const T& v, const Policy&, const boost::true_type&)
{
return v;
}

}

template <class T, class Policy>
inline typename tools::promote_args<T>::type trunc(const T& v, const Policy& pol)
{
return detail::trunc(v, pol, boost::integral_constant<bool, detail::is_integer_for_rounding<T>::value>());
}
template <class T>
inline typename tools::promote_args<T>::type trunc(const T& v)
{
return trunc(v, policies::policy<>());
}
template <class T, class Policy>
inline int itrunc(const T& v, const Policy& pol)
{
BOOST_MATH_STD_USING
typedef typename tools::promote_args<T>::type result_type;
result_type r = boost::math::trunc(v, pol);
if((r > (std::numeric_limits<int>::max)()) || (r < (std::numeric_limits<int>::min)()))
return static_cast<int>(policies::raise_rounding_error("boost::math::itrunc<%1%>(%1%)", 0, static_cast<result_type>(v), 0, pol));
return static_cast<int>(r);
}
template <class T>
inline int itrunc(const T& v)
{
return itrunc(v, policies::policy<>());
}

template <class T, class Policy>
inline long ltrunc(const T& v, const Policy& pol)
{
BOOST_MATH_STD_USING
typedef typename tools::promote_args<T>::type result_type;
result_type r = boost::math::trunc(v, pol);
if((r > (std::numeric_limits<long>::max)()) || (r < (std::numeric_limits<long>::min)()))
return static_cast<long>(policies::raise_rounding_error("boost::math::ltrunc<%1%>(%1%)", 0, static_cast<result_type>(v), 0L, pol));
return static_cast<long>(r);
}
template <class T>
inline long ltrunc(const T& v)
{
return ltrunc(v, policies::policy<>());
}

#ifdef BOOST_HAS_LONG_LONG

template <class T, class Policy>
inline boost::long_long_type lltrunc(const T& v, const Policy& pol)
{
BOOST_MATH_STD_USING
typedef typename tools::promote_args<T>::type result_type;
result_type r = boost::math::trunc(v, pol);
if((r > (std::numeric_limits<boost::long_long_type>::max)()) || (r < (std::numeric_limits<boost::long_long_type>::min)()))
return static_cast<boost::long_long_type>(policies::raise_rounding_error("boost::math::lltrunc<%1%>(%1%)", 0, v, static_cast<boost::long_long_type>(0), pol));
return static_cast<boost::long_long_type>(r);
}
template <class T>
inline boost::long_long_type lltrunc(const T& v)
{
return lltrunc(v, policies::policy<>());
}

#endif

template <class T, class Policy>
inline typename boost::enable_if_c<boost::is_constructible<int, T>::value, int>::type
iconvert(const T& v, const Policy&) 
{
return static_cast<int>(v);
}

template <class T, class Policy>
inline typename boost::disable_if_c<boost::is_constructible<int, T>::value, int>::type
iconvert(const T& v, const Policy& pol) 
{
using boost::math::itrunc;
return itrunc(v, pol);
}

template <class T, class Policy>
inline typename boost::enable_if_c<boost::is_constructible<long, T>::value, long>::type
lconvert(const T& v, const Policy&) 
{
return static_cast<long>(v);
}

template <class T, class Policy>
inline typename boost::disable_if_c<boost::is_constructible<long, T>::value, long>::type
lconvert(const T& v, const Policy& pol) 
{
using boost::math::ltrunc;
return ltrunc(v, pol);
}

#ifdef BOOST_HAS_LONG_LONG

template <class T, class Policy>
inline typename boost::enable_if_c<boost::is_constructible<boost::long_long_type, T>::value, boost::long_long_type>::type
llconvertert(const T& v, const Policy&) 
{
return static_cast<boost::long_long_type>(v);
}

template <class T, class Policy>
inline typename boost::disable_if_c<boost::is_constructible<boost::long_long_type, T>::value, boost::long_long_type>::type
llconvertert(const T& v, const Policy& pol) 
{
using boost::math::lltrunc;
return lltrunc(v, pol);
}

#endif

}} 

#endif 
