


#ifndef BOOST_ACOSH_HPP
#define BOOST_ACOSH_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/config/no_tr1/cmath.hpp>
#include <boost/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/constants/constants.hpp>


namespace boost
{
namespace math
{
namespace detail
{
template<typename T, typename Policy>
inline T    acosh_imp(const T x, const Policy& pol)
{
BOOST_MATH_STD_USING

if((x < 1) || (boost::math::isnan)(x))
{
return policies::raise_domain_error<T>(
"boost::math::acosh<%1%>(%1%)",
"acosh requires x >= 1, but got x = %1%.", x, pol);
}
else if    ((x - 1) >= tools::root_epsilon<T>())
{
if    (x > 1 / tools::root_epsilon<T>())
{
return log(x) + constants::ln_two<T>();
}
else if(x < 1.5f)
{
T y = x - 1;
return boost::math::log1p(y + sqrt(y * y + 2 * y), pol);
}
else
{
return( log( x + sqrt(x * x - 1) ) );
}
}
else
{
T y = x - 1;

T result = sqrt(2 * y) * (1 - y /12 + 3 * y * y / 160);
return result;
}
}
}

template<typename T, typename Policy>
inline typename tools::promote_args<T>::type acosh(T x, const Policy&)
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
detail::acosh_imp(static_cast<value_type>(x), forwarding_policy()),
"boost::math::acosh<%1%>(%1%)");
}
template<typename T>
inline typename tools::promote_args<T>::type acosh(T x)
{
return boost::math::acosh(x, policies::policy<>());
}

}
}

#endif 


