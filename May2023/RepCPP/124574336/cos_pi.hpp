
#ifndef BOOST_MATH_COS_PI_HPP
#define BOOST_MATH_COS_PI_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/config/no_tr1/cmath.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/constants/constants.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
T cos_pi_imp(T x, const Policy& pol)
{
BOOST_MATH_STD_USING 
bool invert = false;
if(fabs(x) < 0.25)
return cos(constants::pi<T>() * x);

if(x < 0)
{
x = -x;
}
T rem = floor(x);
if(iconvert(rem, pol) & 1)
invert = !invert;
rem = x - rem;
if(rem > 0.5f)
{
rem = 1 - rem;
invert = !invert;
}
if(rem == 0.5f)
return 0;

if(rem > 0.25f)
{
rem = 0.5f - rem;
rem = sin(constants::pi<T>() * rem);
}
else
rem = cos(constants::pi<T>() * rem);
return invert ? T(-rem) : rem;
}

} 

template <class T, class Policy>
inline typename tools::promote_args<T>::type cos_pi(T x, const Policy&)
{
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy,
policies::promote_float<false>,
policies::promote_double<false>,
policies::discrete_quantile<>,
policies::assert_undefined<>,
policies::overflow_error<policies::ignore_error> >::type forwarding_policy;
return policies::checked_narrowing_cast<result_type, forwarding_policy>(boost::math::detail::cos_pi_imp<value_type>(x, forwarding_policy()), "cos_pi");
}

template <class T>
inline typename tools::promote_args<T>::type cos_pi(T x)
{
return boost::math::cos_pi(x, policies::policy<>());
}

} 
} 
#endif

