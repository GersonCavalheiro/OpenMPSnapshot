


#ifndef BOOST_ASINH_HPP
#define BOOST_ASINH_HPP

#ifdef _MSC_VER
#pragma once
#endif


#include <boost/config/no_tr1/cmath.hpp>
#include <boost/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/sqrt1pm1.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/constants/constants.hpp>


namespace boost
{
namespace math
{
namespace detail{
template<typename T, class Policy>
inline T    asinh_imp(const T x, const Policy& pol)
{
BOOST_MATH_STD_USING

if((boost::math::isnan)(x))
{
return policies::raise_domain_error<T>(
"boost::math::asinh<%1%>(%1%)",
"asinh requires a finite argument, but got x = %1%.", x, pol);
}
if        (x >= tools::forth_root_epsilon<T>())
{
if        (x > 1 / tools::root_epsilon<T>())
{
return constants::ln_two<T>() + log(x) + 1/ (4 * x * x);
}
else if(x < 0.5f)
{
return boost::math::log1p(x + boost::math::sqrt1pm1(x * x, pol), pol);
}
else
{
return( log( x + sqrt(x*x+1) ) );
}
}
else if    (x <= -tools::forth_root_epsilon<T>())
{
return(-asinh(-x, pol));
}
else
{
T    result = x;

if    (abs(x) >= tools::root_epsilon<T>())
{
T    x3 = x*x*x;

result -= x3/static_cast<T>(6);
}

return(result);
}
}
}

template<typename T>
inline typename tools::promote_args<T>::type asinh(T x)
{
return boost::math::asinh(x, policies::policy<>());
}
template<typename T, typename Policy>
inline typename tools::promote_args<T>::type asinh(T x, const Policy&)
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
detail::asinh_imp(static_cast<value_type>(x), forwarding_policy()),
"boost::math::asinh<%1%>(%1%)");
}

}
}

#endif 

