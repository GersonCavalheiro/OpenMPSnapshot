
#ifndef BOOST_MATH_SF_DETAIL_BESSEL_JY_DERIVATIVES_ASYM_HPP
#define BOOST_MATH_SF_DETAIL_BESSEL_JY_DERIVATIVES_ASYM_HPP

#ifdef _MSC_VER
#pragma once
#endif

namespace boost{ namespace math{ namespace detail{

template <class T>
inline T asymptotic_bessel_derivative_amplitude(T v, T x)
{
BOOST_MATH_STD_USING
T s = 1;
const T mu = 4 * v * v;
T txq = 2 * x;
txq *= txq;

s -= (mu - 3) / (2 * txq);
s -= ((mu - 1) * (mu - 45)) / (txq * txq * 8);

return sqrt(s * 2 / (boost::math::constants::pi<T>() * x));
}

template <class T>
inline T asymptotic_bessel_derivative_phase_mx(T v, T x)
{
const T mu = 4 * v * v;
const T mu2 = mu * mu;
const T mu3 = mu2 * mu;
T denom = 4 * x;
T denom_mult = denom * denom;

T s = 0;
s += (mu + 3) / (2 * denom);
denom *= denom_mult;
s += (mu2 + (46 * mu) - 63) / (6 * denom);
denom *= denom_mult;
s += (mu3 + (185 * mu2) - (2053 * mu) + 1899) / (5 * denom);
return s;
}

template <class T, class Policy>
inline T asymptotic_bessel_y_derivative_large_x_2(T v, T x, const Policy& pol)
{
BOOST_MATH_STD_USING
const T ampl = asymptotic_bessel_derivative_amplitude(v, x);
const T phase = asymptotic_bessel_derivative_phase_mx(v, x);
BOOST_MATH_INSTRUMENT_VARIABLE(ampl);
BOOST_MATH_INSTRUMENT_VARIABLE(phase);
const T cx = cos(x);
const T sx = sin(x);
const T vd2shifted = (v / 2) - 0.25f;
const T ci = cos_pi(vd2shifted, pol);
const T si = sin_pi(vd2shifted, pol);
const T sin_phase = sin(phase) * (cx * ci + sx * si) + cos(phase) * (sx * ci - cx * si);
BOOST_MATH_INSTRUMENT_CODE(sin(phase));
BOOST_MATH_INSTRUMENT_CODE(cos(x));
BOOST_MATH_INSTRUMENT_CODE(cos(phase));
BOOST_MATH_INSTRUMENT_CODE(sin(x));
return sin_phase * ampl;
}

template <class T, class Policy>
inline T asymptotic_bessel_j_derivative_large_x_2(T v, T x, const Policy& pol)
{
BOOST_MATH_STD_USING
const T ampl = asymptotic_bessel_derivative_amplitude(v, x);
const T phase = asymptotic_bessel_derivative_phase_mx(v, x);
BOOST_MATH_INSTRUMENT_VARIABLE(ampl);
BOOST_MATH_INSTRUMENT_VARIABLE(phase);
BOOST_MATH_INSTRUMENT_CODE(cos(phase));
BOOST_MATH_INSTRUMENT_CODE(cos(x));
BOOST_MATH_INSTRUMENT_CODE(sin(phase));
BOOST_MATH_INSTRUMENT_CODE(sin(x));
const T cx = cos(x);
const T sx = sin(x);
const T vd2shifted = (v / 2) - 0.25f;
const T ci = cos_pi(vd2shifted, pol);
const T si = sin_pi(vd2shifted, pol);
const T sin_phase = cos(phase) * (cx * ci + sx * si) - sin(phase) * (sx * ci - cx * si);
BOOST_MATH_INSTRUMENT_VARIABLE(sin_phase);
return sin_phase * ampl;
}

template <class T>
inline bool asymptotic_bessel_derivative_large_x_limit(const T& v, const T& x)
{
BOOST_MATH_STD_USING
return (std::max)(T(fabs(v)), T(1)) < x * sqrt(boost::math::tools::forth_root_epsilon<T>());
}

}}} 

#endif 
