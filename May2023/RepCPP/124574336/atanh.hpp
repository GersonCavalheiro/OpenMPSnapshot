


#ifndef BOOST_ATANH_HPP
#define BOOST_ATANH_HPP

#ifdef _MSC_VER
#pragma once
#endif


#include <boost/config/no_tr1/cmath.hpp>
#include <boost/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/log1p.hpp>


namespace boost
{
namespace math
{
namespace detail
{

template<typename T, typename Policy>
inline T    atanh_imp(const T x, const Policy& pol)
{
BOOST_MATH_STD_USING
static const char* function = "boost::math::atanh<%1%>(%1%)";

if(x < -1)
{
return policies::raise_domain_error<T>(
function,
"atanh requires x >= -1, but got x = %1%.", x, pol);
}
else if(x > 1)
{
return policies::raise_domain_error<T>(
function,
"atanh requires x <= 1, but got x = %1%.", x, pol);
}
else if((boost::math::isnan)(x))
{
return policies::raise_domain_error<T>(
function,
"atanh requires -1 <= x <= 1, but got x = %1%.", x, pol);
}
else if(x < -1 + tools::epsilon<T>())
{
return -policies::raise_overflow_error<T>(function, 0, pol);
}
else if(x > 1 - tools::epsilon<T>())
{
return policies::raise_overflow_error<T>(function, 0, pol);
}
else if(abs(x) >= tools::forth_root_epsilon<T>())
{
if(abs(x) < 0.5f)
return (boost::math::log1p(x, pol) - boost::math::log1p(-x, pol)) / 2;
return(log( (1 + x) / (1 - x) ) / 2);
}
else
{
T    result = x;

if    (abs(x) >= tools::root_epsilon<T>())
{
T    x3 = x*x*x;

result += x3/static_cast<T>(3);
}

return(result);
}
}
}

template<typename T, typename Policy>
inline typename tools::promote_args<T>::type atanh(T x, const Policy&)
{
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;
return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::atanh_imp(static_cast<value_type>(x), forwarding_policy()),
"boost::math::atanh<%1%>(%1%)");
}
template<typename T>
inline typename tools::promote_args<T>::type atanh(T x)
{
return boost::math::atanh(x, policies::policy<>());
}

}
}

#endif 



