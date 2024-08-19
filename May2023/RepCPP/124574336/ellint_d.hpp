
#ifndef BOOST_MATH_ELLINT_D_HPP
#define BOOST_MATH_ELLINT_D_HPP

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
typename tools::promote_args<T1, T2>::type ellint_d(T1 k, T2 phi, const Policy& pol);

namespace detail{

template <typename T, typename Policy>
T ellint_d_imp(T k, const Policy& pol);

template <typename T, typename Policy>
T ellint_d_imp(T phi, T k, const Policy& pol)
{
BOOST_MATH_STD_USING
using namespace boost::math::tools;
using namespace boost::math::constants;

bool invert = false;
if(phi < 0)
{
phi = fabs(phi);
invert = true;
}

T result;

if(phi >= tools::max_value<T>())
{
result = policies::raise_overflow_error<T>("boost::math::ellint_d<%1%>(%1%,%1%)", 0, pol);
}
else if(phi > 1 / tools::epsilon<T>())
{
result = 2 * phi * ellint_d_imp(k, pol) / constants::pi<T>();
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
BOOST_MATH_INSTRUMENT_VARIABLE(rphi);
BOOST_MATH_INSTRUMENT_VARIABLE(m);
T sinp = sin(rphi);
T cosp = cos(rphi);
BOOST_MATH_INSTRUMENT_VARIABLE(sinp);
BOOST_MATH_INSTRUMENT_VARIABLE(cosp);
T c = 1 / (sinp * sinp);
T cm1 = cosp * cosp / (sinp * sinp);  
T k2 = k * k;
if(k2 * sinp * sinp > 1)
{
return policies::raise_domain_error<T>("boost::math::ellint_d<%1%>(%1%, %1%)", "The parameter k is out of range, got k = %1%", k, pol);
}
else if(rphi == 0)
{
result = 0;
}
else
{
result = s * ellint_rd_imp(cm1, T(c - k2), c, pol) / 3;
BOOST_MATH_INSTRUMENT_VARIABLE(result);
}
if(m != 0)
result += m * ellint_d_imp(k, pol);
}
return invert ? T(-result) : result;
}

template <typename T, typename Policy>
T ellint_d_imp(T k, const Policy& pol)
{
BOOST_MATH_STD_USING
using namespace boost::math::tools;

if (abs(k) >= 1)
{
return policies::raise_domain_error<T>("boost::math::ellint_d<%1%>(%1%)",
"Got k = %1%, function requires |k| <= 1", k, pol);
}
if(fabs(k) <= tools::root_epsilon<T>())
return constants::pi<T>() / 4;

T x = 0;
T t = k * k;
T y = 1 - t;
T z = 1;
T value = ellint_rd_imp(x, y, z, pol) / 3;

return value;
}

template <typename T, typename Policy>
inline typename tools::promote_args<T>::type ellint_d(T k, const Policy& pol, const boost::true_type&)
{
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_d_imp(static_cast<value_type>(k), pol), "boost::math::ellint_d<%1%>(%1%)");
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_d(T1 k, T2 phi, const boost::false_type&)
{
return boost::math::ellint_d(k, phi, policies::policy<>());
}

} 

template <typename T>
inline typename tools::promote_args<T>::type ellint_d(T k)
{
return ellint_d(k, policies::policy<>());
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type ellint_d(T1 k, T2 phi)
{
typedef typename policies::is_policy<T2>::type tag_type;
return detail::ellint_d(k, phi, tag_type());
}

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type ellint_d(T1 k, T2 phi, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(detail::ellint_d_imp(static_cast<value_type>(phi), static_cast<value_type>(k), pol), "boost::math::ellint_2<%1%>(%1%,%1%)");
}

}} 

#endif 

