
#ifndef BOOST_MATH_SF_BINOMIAL_HPP
#define BOOST_MATH_SF_BINOMIAL_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/policies/error_handling.hpp>

namespace boost{ namespace math{

template <class T, class Policy>
T binomial_coefficient(unsigned n, unsigned k, const Policy& pol)
{
BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);
BOOST_MATH_STD_USING
static const char* function = "boost::math::binomial_coefficient<%1%>(unsigned, unsigned)";
if(k > n)
return policies::raise_domain_error<T>(
function, 
"The binomial coefficient is undefined for k > n, but got k = %1%.",
static_cast<T>(k), pol);
T result;
if((k == 0) || (k == n))
return static_cast<T>(1);
if((k == 1) || (k == n-1))
return static_cast<T>(n);

if(n <= max_factorial<T>::value)
{
result = unchecked_factorial<T>(n);
result /= unchecked_factorial<T>(n-k);
result /= unchecked_factorial<T>(k);
}
else
{
if(k < n - k)
result = k * beta(static_cast<T>(k), static_cast<T>(n-k+1), pol);
else
result = (n - k) * beta(static_cast<T>(k+1), static_cast<T>(n-k), pol);
if(result == 0)
return policies::raise_overflow_error<T>(function, 0, pol);
result = 1 / result;
}
return ceil(result - 0.5f);
}
template <>
inline float binomial_coefficient<float, policies::policy<> >(unsigned n, unsigned k, const policies::policy<>&)
{
typedef policies::normalise<
policies::policy<>,
policies::promote_float<true>,
policies::promote_double<false>,
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;
return policies::checked_narrowing_cast<float, forwarding_policy>(binomial_coefficient<double>(n, k, forwarding_policy()), "boost::math::binomial_coefficient<%1%>(unsigned,unsigned)");
}

template <class T>
inline T binomial_coefficient(unsigned n, unsigned k)
{
return binomial_coefficient<T>(n, k, policies::policy<>());
}

} 
} 


#endif 



