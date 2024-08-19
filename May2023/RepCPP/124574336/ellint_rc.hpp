
#ifndef BOOST_MATH_ELLINT_RC_HPP
#define BOOST_MATH_ELLINT_RC_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/constants/constants.hpp>
#include <iostream>


namespace boost { namespace math { namespace detail{

template <typename T, typename Policy>
T ellint_rc_imp(T x, T y, const Policy& pol)
{
BOOST_MATH_STD_USING

static const char* function = "boost::math::ellint_rc<%1%>(%1%,%1%)";

if(x < 0)
{
return policies::raise_domain_error<T>(function,
"Argument x must be non-negative but got %1%", x, pol);
}
if(y == 0)
{
return policies::raise_domain_error<T>(function,
"Argument y must not be zero but got %1%", y, pol);
}

T prefix, result;
if(y < 0)
{
prefix = sqrt(x / (x - y));
x = x - y;
y = -y;
}
else
prefix = 1;

if(x == 0)
{
result = constants::half_pi<T>() / sqrt(y);
}
else if(x == y)
{
result = 1 / sqrt(x);
}
else if(y > x)
{
result = atan(sqrt((y - x) / x)) / sqrt(y - x);
}
else
{
if(y / x > 0.5)
{
T arg = sqrt((x - y) / x);
result = (boost::math::log1p(arg, pol) - boost::math::log1p(-arg, pol)) / (2 * sqrt(x - y));
}
else
{
result = log((sqrt(x) + sqrt(x - y)) / sqrt(y)) / sqrt(x - y);
}
}
return prefix * result;
}

} 

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type 
ellint_rc(T1 x, T2 y, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(
detail::ellint_rc_imp(
static_cast<value_type>(x),
static_cast<value_type>(y), pol), "boost::math::ellint_rc<%1%>(%1%,%1%)");
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type 
ellint_rc(T1 x, T2 y)
{
return ellint_rc(x, y, policies::policy<>());
}

}} 

#endif 

