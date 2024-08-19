


#ifndef BOOST_SINHC_HPP
#define BOOST_SINHC_HPP


#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/precision.hpp>
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
inline T    sinhc_pi_imp(const T x)
{
#if defined(BOOST_NO_STDC_NAMESPACE) && !defined(__SUNPRO_CC)
using    ::abs;
using    ::sinh;
using    ::sqrt;
#else    
using    ::std::abs;
using    ::std::sinh;
using    ::std::sqrt;
#endif    

static T const    taylor_0_bound = tools::epsilon<T>();
static T const    taylor_2_bound = sqrt(taylor_0_bound);
static T const    taylor_n_bound = sqrt(taylor_2_bound);

if    (abs(x) >= taylor_n_bound)
{
return(sinh(x)/x);
}
else
{
T    result = static_cast<T>(1);

if    (abs(x) >= taylor_0_bound)
{
T    x2 = x*x;

result += x2/static_cast<T>(6);

if    (abs(x) >= taylor_2_bound)
{
result += (x2*x2)/static_cast<T>(120);
}
}

return(result);
}
}

} 

template <class T>
inline typename tools::promote_args<T>::type sinhc_pi(T x)
{
typedef typename tools::promote_args<T>::type result_type;
return detail::sinhc_pi_imp(static_cast<result_type>(x));
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type sinhc_pi(T x, const Policy&)
{
return boost::math::sinhc_pi(x);
}

#ifdef    BOOST_NO_TEMPLATE_TEMPLATES
#else    
template<typename T, template<typename> class U>
inline U<T>    sinhc_pi(const U<T> x)
{
#if defined(BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL) || defined(__GNUC__)
using namespace std;
#elif    defined(BOOST_NO_STDC_NAMESPACE) && !defined(__SUNPRO_CC)
using    ::abs;
using    ::sinh;
using    ::sqrt;
#else    
using    ::std::abs;
using    ::std::sinh;
using    ::std::sqrt;
#endif    

using    ::std::numeric_limits;

static T const    taylor_0_bound = tools::epsilon<T>();
static T const    taylor_2_bound = sqrt(taylor_0_bound);
static T const    taylor_n_bound = sqrt(taylor_2_bound);

if    (abs(x) >= taylor_n_bound)
{
return(sinh(x)/x);
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

result += x2/static_cast<T>(6);

if    (abs(x) >= taylor_2_bound)
{
result += (x2*x2)/static_cast<T>(120);
}
}

return(result);
}
}
#endif    
}
}

#endif 

