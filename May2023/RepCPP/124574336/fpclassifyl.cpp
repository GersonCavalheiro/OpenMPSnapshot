#  include <pch.hpp>
#ifndef BOOST_MATH_TR1_SOURCE
#  define BOOST_MATH_TR1_SOURCE
#endif
#include <boost/math/tr1.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/sign.hpp>
#include "c_policy.hpp"

#if defined (_MSC_VER)
#  pragma warning(push)
#  pragma warning (disable: 4800) 
#endif

namespace boost{ namespace math{ namespace tr1{

template<> bool BOOST_MATH_TR1_DECL signbit<long double> BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{
return static_cast<bool>((boost::math::signbit)(x));
}

template<> int BOOST_MATH_TR1_DECL fpclassify<long double> BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{
return (boost::math::fpclassify)(x);
}

template<> bool BOOST_MATH_TR1_DECL isfinite<long double> BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{
return (boost::math::isfinite)(x);
}

template<> bool BOOST_MATH_TR1_DECL isinf<long double> BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{
return (boost::math::isinf)(x);
}

template<> bool BOOST_MATH_TR1_DECL isnan<long double> BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{
return (boost::math::isnan)(x);
}

template<> bool BOOST_MATH_TR1_DECL isnormal<long double> BOOST_PREVENT_MACRO_SUBSTITUTION(long double x)
{
return (boost::math::isnormal)(x);
}

}}} 

#if defined (_MSC_VER)
#  pragma warning(pop)
#endif

