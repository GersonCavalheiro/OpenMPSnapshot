
#ifndef BOOST_MATH_SF_DETAIL_BESSEL_JY_ASYM_HPP
#define BOOST_MATH_SF_DETAIL_BESSEL_JY_ASYM_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/factorials.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T>
inline T asymptotic_bessel_amplitude(T v, T x)
{
BOOST_MATH_STD_USING
T s = 1;
T mu = 4 * v * v;
T txq = 2 * x;
txq *= txq;

s += (mu - 1) / (2 * txq);
s += 3 * (mu - 1) * (mu - 9) / (txq * txq * 8);
s += 15 * (mu - 1) * (mu - 9) * (mu - 25) / (txq * txq * txq * 8 * 6);

return sqrt(s * 2 / (constants::pi<T>() * x));
}

template <class T>
T asymptotic_bessel_phase_mx(T v, T x)
{
T mu = 4 * v * v;
T denom = 4 * x;
T denom_mult = denom * denom;

T s = 0;
s += (mu - 1) / (2 * denom);
denom *= denom_mult;
s += (mu - 1) * (mu - 25) / (6 * denom);
denom *= denom_mult;
s += (mu - 1) * (mu * mu - 114 * mu + 1073) / (5 * denom);
denom *= denom_mult;
s += (mu - 1) * (5 * mu * mu * mu - 1535 * mu * mu + 54703 * mu - 375733) / (14 * denom);
return s;
}

template <class T, class Policy>
inline T asymptotic_bessel_y_large_x_2(T v, T x, const Policy& pol)
{
BOOST_MATH_STD_USING
T ampl = asymptotic_bessel_amplitude(v, x);
T phase = asymptotic_bessel_phase_mx(v, x);
BOOST_MATH_INSTRUMENT_VARIABLE(ampl);
BOOST_MATH_INSTRUMENT_VARIABLE(phase);
T cx = cos(x);
T sx = sin(x);
T ci = boost::math::cos_pi(v / 2 + 0.25f, pol);
T si = boost::math::sin_pi(v / 2 + 0.25f, pol);
T sin_phase = sin(phase) * (cx * ci + sx * si) + cos(phase) * (sx * ci - cx * si);
BOOST_MATH_INSTRUMENT_CODE(sin(phase));
BOOST_MATH_INSTRUMENT_CODE(cos(x));
BOOST_MATH_INSTRUMENT_CODE(cos(phase));
BOOST_MATH_INSTRUMENT_CODE(sin(x));
return sin_phase * ampl;
}

template <class T, class Policy>
inline T asymptotic_bessel_j_large_x_2(T v, T x, const Policy& pol)
{
BOOST_MATH_STD_USING
T ampl = asymptotic_bessel_amplitude(v, x);
T phase = asymptotic_bessel_phase_mx(v, x);
BOOST_MATH_INSTRUMENT_VARIABLE(ampl);
BOOST_MATH_INSTRUMENT_VARIABLE(phase);
BOOST_MATH_INSTRUMENT_CODE(cos(phase));
BOOST_MATH_INSTRUMENT_CODE(cos(x));
BOOST_MATH_INSTRUMENT_CODE(sin(phase));
BOOST_MATH_INSTRUMENT_CODE(sin(x));
T cx = cos(x);
T sx = sin(x);
T ci = boost::math::cos_pi(v / 2 + 0.25f, pol);
T si = boost::math::sin_pi(v / 2 + 0.25f, pol);
T sin_phase = cos(phase) * (cx * ci + sx * si) - sin(phase) * (sx * ci - cx * si);
BOOST_MATH_INSTRUMENT_VARIABLE(sin_phase);
return sin_phase * ampl;
}

template <class T>
inline bool asymptotic_bessel_large_x_limit(int v, const T& x)
{
BOOST_MATH_STD_USING
BOOST_ASSERT(v >= 0);
return (v ? v : 1) < x * 0.004f;
}

template <class T>
inline bool asymptotic_bessel_large_x_limit(const T& v, const T& x)
{
BOOST_MATH_STD_USING
return (std::max)(T(fabs(v)), T(1)) < x * sqrt(tools::forth_root_epsilon<T>());
}

template <class T, class Policy>
void temme_asyptotic_y_small_x(T v, T x, T* Y, T* Y1, const Policy& pol)
{
T c = 1;
T p = (v / boost::math::sin_pi(v, pol)) * pow(x / 2, -v) / boost::math::tgamma(1 - v, pol);
T q = (v / boost::math::sin_pi(v, pol)) * pow(x / 2, v) / boost::math::tgamma(1 + v, pol);
T f = (p - q) / v;
T g_prefix = boost::math::sin_pi(v / 2, pol);
g_prefix *= g_prefix * 2 / v;
T g = f + g_prefix * q;
T h = p;
T c_mult = -x * x / 4;

T y(c * g), y1(c * h);

for(int k = 1; k < policies::get_max_series_iterations<Policy>(); ++k)
{
f = (k * f + p + q) / (k*k - v*v);
p /= k - v;
q /= k + v;
c *= c_mult / k;
T c1 = pow(-x * x / 4, k) / factorial<T>(k, pol);
g = f + g_prefix * q;
h = -k * g + p;
y += c * g;
y1 += c * h;
if(c * g / tools::epsilon<T>() < y)
break;
}

*Y = -y;
*Y1 = (-2 / x) * y1;
}

template <class T, class Policy>
T asymptotic_bessel_i_large_x(T v, T x, const Policy& pol)
{
BOOST_MATH_STD_USING  
T s = 1;
T mu = 4 * v * v;
T ex = 8 * x;
T num = mu - 1;
T denom = ex;

s -= num / denom;

num *= mu - 9;
denom *= ex * 2;
s += num / denom;

num *= mu - 25;
denom *= ex * 3;
s -= num / denom;

T e = exp(x/2);

s = e * (e * s / sqrt(2 * x * constants::pi<T>()));

return (boost::math::isfinite)(s) ? 
s : policies::raise_overflow_error<T>("boost::math::asymptotic_bessel_i_large_x<%1%>(%1%,%1%)", 0, pol);
}

}}} 

#endif

