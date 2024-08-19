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

template<> bool BOOST_MATH_TR1_DECL signbit<float> BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{
return static_cast<bool>((boost::math::signbit)(x));
}

template<> int BOOST_MATH_TR1_DECL fpclassify<float> BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{
return (boost::math::fpclassify)(x);
}

template<> bool BOOST_MATH_TR1_DECL isfinite<float> BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{
return (boost::math::isfinite)(x);
}

template<> bool BOOST_MATH_TR1_DECL isinf<float> BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{
return (boost::math::isinf)(x);
}

template<> bool BOOST_MATH_TR1_DECL isnan<float> BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{
return (boost::math::isnan)(x);
}

template<> bool BOOST_MATH_TR1_DECL isnormal<float> BOOST_PREVENT_MACRO_SUBSTITUTION(float x)
{
return (boost::math::isnormal)(x);
}

}}} 

#if defined (_MSC_VER)
#  pragma warning(pop)
#endif


