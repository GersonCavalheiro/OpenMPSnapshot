
#ifndef BOOST_MATH_ELLINT_3_HPP
#define BOOST_MATH_ELLINT_3_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/atanh.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/workaround.hpp>
#include <boost/math/special_functions/round.hpp>


namespace boost { namespace math { 

namespace detail{

template <typename T, typename Policy>
T ellint_pi_imp(T v, T k, T vc, const Policy& pol);

template <typename T, typename Policy>
T ellint_pi_imp(T v, T phi, T k, T vc, const Policy& pol)
{
BOOST_MATH_STD_USING

static const char* function = "boost::math::ellint_3<%1%>(%1%,%1%,%1%)";


T sphi = sin(fabs(phi));
T result = 0;

if (k * k * sphi * sphi > 1)
{
return policies::raise_domain_error<T>(function,
"Got k = %1%, function requires |k| <= 1", k, pol);
}
if(v == 0)
{
return (k == 0) ? phi : ellint_f_imp(phi, k, pol);
}
if((v > 0) && (1 / v < (sphi * sphi)))
{
return policies::raise_domain_error<T>(function,
"Got v = %1%, but result is complex for v > 1 / sin^2(phi)", v, pol);
}

if(v == 1)
{
if (k == 0)
return tan(phi);

T m = k * k;
result = sqrt(1 - m * sphi * sphi) * tan(phi) - ellint_e_imp(phi, k, pol);
result /= 1 - m;
result += ellint_f_imp(phi, k, pol);
return result;
}
if(phi == constants::half_pi<T>())
{
return ellint_pi_imp(v, k, vc, pol);
}
if((phi > constants::half_pi<T>()) || (phi < 0))
{
if(fabs(phi) > 1 / tools::epsilon<T>())
{
if(v > 1)
return policies::raise_domain_error<T>(
function,
"Got v = %1%, but this is only supported for 0 <= phi <= pi/2", v, pol);
result = 2 * fabs(phi) * ellint_pi_imp(v, k, vc, pol) / constants::pi<T>();
}
else
{
T rphi = boost::math::tools::fmod_workaround(T(fabs(phi)), T(constants::half_pi<T>()));
T m = boost::math::round((fabs(phi) - rphi) / constants::half_pi<T>());
int sign = 1;
if((m != 0) && (k >= 1))
{
return policies::raise_domain_error<T>(function, "Got k=1 and phi=%1% but the result is complex in that domain", phi, pol);
}
if(boost::math::tools::fmod_workaround(m, T(2)) > 0.5)
{
m += 1;
sign = -1;
rphi = constants::half_pi<T>() - rphi;
}
result = sign * ellint_pi_imp(v, rphi, k, vc, pol);
if((m > 0) && (vc > 0))
result += m * ellint_pi_imp(v, k, vc, pol);
}
return phi < 0 ? T(-result) : result;
}
if(k == 0)
{
if(v < 1)
{
T vcr = sqrt(vc);
return atan(vcr * tan(phi)) / vcr;
}
else
{
T vcr = sqrt(-vc);
T arg = vcr * tan(phi);
return (boost::math::log1p(arg, pol) - boost::math::log1p(-arg, pol)) / (2 * vcr);
}
}
if((v < 0) && fabs(k) <= 1)
{
T k2 = k * k;
T N = (k2 - v) / (1 - v);
T Nm1 = (1 - k2) / (1 - v);
T p2 = -v * N;
T t;
if(p2 <= tools::min_value<T>())
p2 = sqrt(-v) * sqrt(N);
else
p2 = sqrt(p2);
T delta = sqrt(1 - k2 * sphi * sphi);
if(N > k2)
{
result = ellint_pi_imp(N, phi, k, Nm1, pol);
result *= v / (v - 1);
result *= (k2 - 1) / (v - k2);
}

if(k != 0)
{
t = ellint_f_imp(phi, k, pol);
t *= k2 / (k2 - v);
result += t;
}
t = v / ((k2 - v) * (v - 1));
if(t > tools::min_value<T>())
{
result += atan((p2 / 2) * sin(2 * phi) / delta) * sqrt(t);
}
else
{
result += atan((p2 / 2) * sin(2 * phi) / delta) * sqrt(fabs(1 / (k2 - v))) * sqrt(fabs(v / (v - 1)));
}
return result;
}
if(k == 1)
{
result = sqrt(v) * atanh(sqrt(v) * sin(phi), pol) - log(1 / cos(phi) + tan(phi));
result /= v - 1;
return result;
}
#if 0  
if(v > 1)
{
T k2 = k * k;
T N = k2 / v;
T Nm1 = (v - k2) / v;
T p1 = sqrt((-vc) * (1 - k2 / v));
T delta = sqrt(1 - k2 * sphi * sphi);
result = -ellint_pi_imp(N, phi, k, Nm1, pol);
result += ellint_f_imp(phi, k, pol);
result += log((delta + p1 * tan(phi)) / (delta - p1 * tan(phi))) / (2 * p1);
return result;
}
#endif
BOOST_ASSERT(fabs(phi) < constants::half_pi<T>());
BOOST_ASSERT(phi >= 0);
T x, y, z, p, t;
T cosp = cos(phi);
x = cosp * cosp;
t = sphi * sphi;
y = 1 - k * k * t;
z = 1;
if(v * t < 0.5)
p = 1 - v * t;
else
p = x + vc * t;
result = sphi * (ellint_rf_imp(x, y, z, pol) + v * t * ellint_rj_imp(x, y, z, p, pol) / 3);

return result;
}

template <typename T, typename Policy>
T ellint_pi_imp(T v, T k, T vc, const Policy& pol)
{
BOOST_MATH_STD_USING
using namespace boost::math::tools;

static const char* function = "boost::math::ellint_pi<%1%>(%1%,%1%)";

if (abs(k) >= 1)
{
return policies::raise_domain_error<T>(function,
"Got k = %1%, function requires |k| <= 1", k, pol);
}
if(vc <= 0)
{
return policies::raise_domain_error<T>(function,
"Got v = %1%, function requires v < 1", v, pol);
}

if(v == 0)
{
return (k == 0) ? boost::math::constants::pi<T>() / 2 : ellint_k_imp(k, pol);
}

if(v < 0)
{
T k2 = k * k;
T N = (k2 - v) / (1 - v);
T Nm1 = (1 - k2) / (1 - v);
T result = 0;
result = boost::math::detail::ellint_pi_imp(N, k, Nm1, pol);
result *= -v / (1 - v);
result *= (1 - k2) / (k2 - v);
result += ellint_k_imp(k, pol) * k2 / (k2 - v);
return result;
}

T x = 0;
T y = 1 - k * k;
T z = 1;
T p = vc;
T value = ellint_rf_imp(x, y, z, pol) + v * ellint_rj_imp(x, y, z, p, pol) / 3;

return value;
}

template <class T1, class T2, class T3>
inline typename tools::promote_args<T1, T2, T3>::type ellint_3(T1 k, T2 v, T3 phi, const boost::false_type&)
{
return boost::math::ellint_3(k, v, phi, policies::policy<>());
}

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type ellint_3(T1 k, T2 v, const Policy& pol, const boost::true_type&)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(
detail::ellint_pi_imp(
static_cast<value_type>(v), 
static_cast<value_type>(k),
static_cast<value_type>(1-v),
pol), "boost::math::ellint_3<%1%>(%1%,%1%)");
}

} 

template <class T1, class T2, class T3, class Policy>
inline typename tools::promote_args<T1, T2, T3>::type ellint_3(T1 k, T2 v, T3 phi, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2, T3>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(
detail::ellint_pi_imp(
static_cast<value_type>(v), 
static_cast<value_type>(phi), 
static_cast<value_type>(k),
static_cast<value_type>(1-v),
pol), "boost::math::ellint_3<%1%>(%1%,%1%,%1%)");
}

template <class T1, class T2, class T3>
typename detail::ellint_3_result<T1, T2, T3>::type ellint_3(T1 k, T2 v, T3 phi)
{
typedef typename policies::is_policy<T3>::type tag_type;
return detail::ellint_3(k, v, phi, tag_type());
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_3(T1 k, T2 v)
{
return ellint_3(k, v, policies::policy<>());
}

}} 

#endif 

