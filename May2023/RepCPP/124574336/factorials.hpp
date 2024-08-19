
#ifndef BOOST_MATH_SP_FACTORIALS_HPP
#define BOOST_MATH_SP_FACTORIALS_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/detail/unchecked_factorial.hpp>
#include <boost/array.hpp>
#ifdef BOOST_MSVC
#pragma warning(push) 
#pragma warning(disable: 4127 4701)
#endif
#ifdef BOOST_MSVC
#pragma warning(pop)
#endif
#include <boost/config/no_tr1/cmath.hpp>

namespace boost { namespace math
{

template <class T, class Policy>
inline T factorial(unsigned i, const Policy& pol)
{
BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);

BOOST_MATH_STD_USING 

if(i <= max_factorial<T>::value)
return unchecked_factorial<T>(i);
T result = boost::math::tgamma(static_cast<T>(i+1), pol);
if(result > tools::max_value<T>())
return result; 
return floor(result + 0.5f);
}

template <class T>
inline T factorial(unsigned i)
{
return factorial<T>(i, policies::policy<>());
}

template <class T, class Policy>
T double_factorial(unsigned i, const Policy& pol)
{
BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);
BOOST_MATH_STD_USING  
if(i & 1)
{
if(i < max_factorial<T>::value)
{
unsigned n = (i - 1) / 2;
return ceil(unchecked_factorial<T>(i) / (ldexp(T(1), (int)n) * unchecked_factorial<T>(n)) - 0.5f);
}
T result = boost::math::tgamma(static_cast<T>(i) / 2 + 1, pol) / sqrt(constants::pi<T>());
if(ldexp(tools::max_value<T>(), -static_cast<int>(i+1) / 2) > result)
return ceil(result * ldexp(T(1), static_cast<int>(i+1) / 2) - 0.5f);
}
else
{
unsigned n = i / 2;
T result = factorial<T>(n, pol);
if(ldexp(tools::max_value<T>(), -(int)n) > result)
return result * ldexp(T(1), (int)n);
}
return policies::raise_overflow_error<T>("boost::math::double_factorial<%1%>(unsigned)", 0, pol);
}

template <class T>
inline T double_factorial(unsigned i)
{
return double_factorial<T>(i, policies::policy<>());
}

namespace detail{

template <class T, class Policy>
T rising_factorial_imp(T x, int n, const Policy& pol)
{
BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);
if(x < 0)
{
bool inv = false;
if(n < 0)
{
x += n;
n = -n;
inv = true;
}
T result = ((n&1) ? -1 : 1) * falling_factorial(-x, n, pol);
if(inv)
result = 1 / result;
return result;
}
if(n == 0)
return 1;
if(x == 0)
{
if(n < 0)
return -boost::math::tgamma_delta_ratio(x + 1, static_cast<T>(-n), pol);
else
return 0;
}
if((x < 1) && (x + n < 0))
{
T val = boost::math::tgamma_delta_ratio(1 - x, static_cast<T>(-n), pol);
return (n & 1) ? T(-val) : val;
}
return 1 / boost::math::tgamma_delta_ratio(x, static_cast<T>(n), pol);
}

template <class T, class Policy>
inline T falling_factorial_imp(T x, unsigned n, const Policy& pol)
{
BOOST_STATIC_ASSERT(!boost::is_integral<T>::value);
BOOST_MATH_STD_USING 
if(x == 0)
return 0;
if(x < 0)
{
return (n&1 ? -1 : 1) * rising_factorial(-x, n, pol);
}
if(n == 0)
return 1;
if(x < 0.5f)
{
if(n > max_factorial<T>::value - 2)
{
T t1 = x * boost::math::falling_factorial(x - 1, max_factorial<T>::value - 2, pol);
T t2 = boost::math::falling_factorial(x - max_factorial<T>::value + 1, n - max_factorial<T>::value + 1, pol);
if(tools::max_value<T>() / fabs(t1) < fabs(t2))
return boost::math::sign(t1) * boost::math::sign(t2) * policies::raise_overflow_error<T>("boost::math::falling_factorial<%1%>", 0, pol);
return t1 * t2;
}
return x * boost::math::falling_factorial(x - 1, n - 1, pol);
}
if(x <= n - 1)
{
T xp1 = x + 1;
unsigned n2 = itrunc((T)floor(xp1), pol);
if(n2 == xp1)
return 0;
T result = boost::math::tgamma_delta_ratio(xp1, -static_cast<T>(n2), pol);
x -= n2;
result *= x;
++n2;
if(n2 < n)
result *= falling_factorial(x - 1, n - n2, pol);
return result;
}
return boost::math::tgamma_delta_ratio(x + 1, -static_cast<T>(n), pol);
}

} 

template <class RT>
inline typename tools::promote_args<RT>::type
falling_factorial(RT x, unsigned n)
{
typedef typename tools::promote_args<RT>::type result_type;
return detail::falling_factorial_imp(
static_cast<result_type>(x), n, policies::policy<>());
}

template <class RT, class Policy>
inline typename tools::promote_args<RT>::type
falling_factorial(RT x, unsigned n, const Policy& pol)
{
typedef typename tools::promote_args<RT>::type result_type;
return detail::falling_factorial_imp(
static_cast<result_type>(x), n, pol);
}

template <class RT>
inline typename tools::promote_args<RT>::type
rising_factorial(RT x, int n)
{
typedef typename tools::promote_args<RT>::type result_type;
return detail::rising_factorial_imp(
static_cast<result_type>(x), n, policies::policy<>());
}

template <class RT, class Policy>
inline typename tools::promote_args<RT>::type
rising_factorial(RT x, int n, const Policy& pol)
{
typedef typename tools::promote_args<RT>::type result_type;
return detail::rising_factorial_imp(
static_cast<result_type>(x), n, pol);
}

} 
} 

#endif 

