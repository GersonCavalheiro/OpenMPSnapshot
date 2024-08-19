#ifndef BOOST_MATH_ELLINT_RF_HPP
#define BOOST_MATH_ELLINT_RF_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/ellint_rc.hpp>


namespace boost { namespace math { namespace detail{

template <typename T, typename Policy>
T ellint_rf_imp(T x, T y, T z, const Policy& pol)
{
BOOST_MATH_STD_USING
using namespace boost::math;
using std::swap;

static const char* function = "boost::math::ellint_rf<%1%>(%1%,%1%,%1%)";

if(x < 0 || y < 0 || z < 0)
{
return policies::raise_domain_error<T>(function,
"domain error, all arguments must be non-negative, "
"only sensible result is %1%.",
std::numeric_limits<T>::quiet_NaN(), pol);
}
if(x + y == 0 || y + z == 0 || z + x == 0)
{
return policies::raise_domain_error<T>(function,
"domain error, at most one argument can be zero, "
"only sensible result is %1%.",
std::numeric_limits<T>::quiet_NaN(), pol);
}
if(x == y)
{
if(x == z)
{
return 1 / sqrt(x);
}
else
{
if(z == 0)
return constants::pi<T>() / (2 * sqrt(x));
else
return ellint_rc_imp(z, x, pol);
}
}
if(x == z)
{
if(y == 0)
return constants::pi<T>() / (2 * sqrt(x));
else
return ellint_rc_imp(y, x, pol);
}
if(y == z)
{
if(x == 0)
return constants::pi<T>() / (2 * sqrt(y));
else
return ellint_rc_imp(x, y, pol);
}
if(x == 0)
swap(x, z);
else if(y == 0)
swap(y, z);
if(z == 0)
{
T xn = sqrt(x);
T yn = sqrt(y);

while(fabs(xn - yn) >= 2.7 * tools::root_epsilon<T>() * fabs(xn))
{
T t = sqrt(xn * yn);
xn = (xn + yn) / 2;
yn = t;
}
return constants::pi<T>() / (xn + yn);
}

T xn = x;
T yn = y;
T zn = z;
T An = (x + y + z) / 3;
T A0 = An;
T Q = pow(3 * boost::math::tools::epsilon<T>(), T(-1) / 8) * (std::max)((std::max)(fabs(An - xn), fabs(An - yn)), fabs(An - zn));
T fn = 1;


unsigned k = 1;
for(; k < boost::math::policies::get_max_series_iterations<Policy>(); ++k)
{
T root_x = sqrt(xn);
T root_y = sqrt(yn);
T root_z = sqrt(zn);
T lambda = root_x * root_y + root_x * root_z + root_y * root_z;
An = (An + lambda) / 4;
xn = (xn + lambda) / 4;
yn = (yn + lambda) / 4;
zn = (zn + lambda) / 4;
Q /= 4;
fn *= 4;
if(Q < fabs(An))
break;
}
policies::check_series_iterations<T>(function, k, pol);
BOOST_MATH_INSTRUMENT_VARIABLE(k);

T X = (A0 - x) / (An * fn);
T Y = (A0 - y) / (An * fn);
T Z = -X - Y;

T E2 = X * Y - Z * Z;
T E3 = X * Y * Z;
return (1 + E3 * (T(1) / 14 + 3 * E3 / 104) + E2 * (T(-1) / 10 + E2 / 24 - (3 * E3) / 44 - 5 * E2 * E2 / 208 + E2 * E3 / 16)) / sqrt(An);
}

} 

template <class T1, class T2, class T3, class Policy>
inline typename tools::promote_args<T1, T2, T3>::type 
ellint_rf(T1 x, T2 y, T3 z, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2, T3>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
return policies::checked_narrowing_cast<result_type, Policy>(
detail::ellint_rf_imp(
static_cast<value_type>(x),
static_cast<value_type>(y),
static_cast<value_type>(z), pol), "boost::math::ellint_rf<%1%>(%1%,%1%,%1%)");
}

template <class T1, class T2, class T3>
inline typename tools::promote_args<T1, T2, T3>::type 
ellint_rf(T1 x, T2 y, T3 z)
{
return ellint_rf(x, y, z, policies::policy<>());
}

}} 

#endif 

