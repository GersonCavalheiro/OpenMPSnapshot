


#ifndef BOOST_SINC_HPP
#define BOOST_SINC_HPP


#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/config/no_tr1/cmath.hpp>
#include <boost/limits.hpp>
#include <string>
#include <stdexcept>


#include <boost/config.hpp>



namespace boost
{
namespace math
{
namespace detail
{

template<typename T>
inline T    sinc_pi_imp(const T x)
{
BOOST_MATH_STD_USING

if    (abs(x) >= 3.3 * tools::forth_root_epsilon<T>())
{
return(sin(x)/x);
}
else
{
return 1 - x * x / 6;
}
}

} 

template <class T>
inline typename tools::promote_args<T>::type sinc_pi(T x)
{
typedef typename tools::promote_args<T>::type result_type;
return detail::sinc_pi_imp(static_cast<result_type>(x));
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type sinc_pi(T x, const Policy&)
{
typedef typename tools::promote_args<T>::type result_type;
return detail::sinc_pi_imp(static_cast<result_type>(x));
}

#ifndef    BOOST_NO_TEMPLATE_TEMPLATES
template<typename T, template<typename> class U>
inline U<T>    sinc_pi(const U<T> x)
{
BOOST_MATH_STD_USING
using    ::std::numeric_limits;

T const    taylor_0_bound = tools::epsilon<T>();
T const    taylor_2_bound = tools::root_epsilon<T>();
T const    taylor_n_bound = tools::forth_root_epsilon<T>();

if    (abs(x) >= taylor_n_bound)
{
return(sin(x)/x);
}
else
{
#ifdef __MWERKS__
U<T>    result = static_cast<U<T> >(1);
#else
U<T>    result = U<T>(1);
#endif

if    (abs(x) >= taylor_0_bound)
{
U<T>    x2 = x*x;

result -= x2/static_cast<T>(6);

if    (abs(x) >= taylor_2_bound)
{
result += (x2*x2)/static_cast<T>(120);
}
}

return(result);
}
}

template<typename T, template<typename> class U, class Policy>
inline U<T>    sinc_pi(const U<T> x, const Policy&)
{
return sinc_pi(x);
}
#endif    
}
}

#endif 

