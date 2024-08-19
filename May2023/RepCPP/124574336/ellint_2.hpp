
#ifndef BOOST_MATH_ELLINT_2_HPP
#define BOOST_MATH_ELLINT_2_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <boost/math/special_functions/ellint_rg.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/workaround.hpp>
#include <boost/math/special_functions/round.hpp>


namespace boost { namespace math { 

template <class T1, class T2, class Policy>
typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi, const Policy& pol);

namespace detail{

template <typename T, typename Policy>
T ellint_e_imp(T k, const Policy& pol);

template <typename T, typename Policy>
T ellint_e_imp(T phi, T k, const Policy& pol)
{
BOOST_MATH_STD_USING
using namespace boost::math::tools;
using namespace boost::math::constants;

bool invert = false;
if (phi == 0)
return 0;

if(phi < 0)
{
phi = fabs(phi);
invert = true;
}

T result;

if(phi >= tools::max_value<T>())
{
result = policies::raise_overflow_error<T>("boost::math::ellint_e<%1%>(%1%,%1%)", 0, pol);
}
else if(phi > 1 / tools::epsilon<T>())
{
result = 2 * phi * ellint_e_imp(k, pol) / constants::pi<T>();
}
else if(k == 0)
{
return invert ? T(-phi) : phi;
}
else if(fabs(k) == 1)
{
T m = boost::math::round(phi / boost::math::constants::pi<T>());
T remains = phi - m * boost::math::constants::pi<T>();
T value = 2 * m + sin(remains);

return invert ? -value : value;
}
else
{
T rphi = boost::math::tools::fmod_workaround(phi, T(constants::half_pi<T>()));
T m = boost::math::round((phi - rphi) / constants::half_pi<T>());
int s = 1;
if(boost::math::tools::fmod_workaround(m, T(2)) > 0.5)
{
m += 1;
s = -1;
rphi = constants::half_pi<T>() - rphi;
}
T k2 = k * k;
if(boost::math::pow<3>(rphi) * k2 / 6 < tools::epsilon<T>() * fabs(rphi))
{
result = s * rphi;
}
else
{
T sinp = sin(rphi);
if (k2 * sinp * sinp >= 1)
{
return policies::raise_domain_error<T>("boost::math::ellint_2<%1%>(%1%, %1%)", "The parameter k is out of range, got k = %1%", k, pol);
}
T cosp = cos(rphi);
T c = 1 / (sinp * sinp);
T cm1 = cosp * cosp / (sinp * sinp);  
result = s * ((1 - k2) * ellint_rf_imp(cm1, T(c - k2), c, pol) + k2 * (1 - k2) * ellint_rd(cm1, c, T(c - k2), pol) / 3 + k2 * sqrt(cm1 / (c * (c - k2))));
}
if(m != 0)
result += m * ellint_e_imp(k, pol);
}
return invert ? T(-result) : result;
}

template <typename T, typename Policy>
T ellint_e_imp(T k, const Policy& pol)
{
BOOST_MATH_STD_USING
using namespace boost::math::tools;

if (abs(k) > 1)
{
return policies::raise_domain_error<T>("boost::math::ellint_e<%1%>(%1%)",
"Got k = %1%, function requires |k| <= 1", k, pol);
}
if (abs(k) == 1)
{
return static_cast<T>(1);
}

T x = 0;
T t = k * k;
T y = 1 - t;
T z = 1;
T value = 2 * ellint_rg_imp(x, y, z, pol);

return value;
}

template <typename T, typename Policy>
inline typename tools::promote_args<T>::type ellint_2(T k, const Policy& pol, const boost::true_type&)
{
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_e_imp(static_cast<value_type>(k), pol), "boost::math::ellint_2<%1%>(%1%)");
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi, const boost::false_type&)
{
return boost::math::ellint_2(k, phi, policies::policy<>());
}

} 

template <typename T>
inline typename tools::promote_args<T>::type ellint_2(T k)
{
return ellint_2(k, policies::policy<>());
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi)
{
typedef typename policies::is_policy<T2>::type tag_type;
return detail::ellint_2(k, phi, tag_type());
}

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type ellint_2(T1 k, T2 phi, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_e_imp(static_cast<value_type>(phi), static_cast<value_type>(k), pol), "boost::math::ellint_2<%1%>(%1%,%1%)");
}

}} 

#endif 

