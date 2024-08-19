

#ifndef BOOST_MATH_TOOLS_MINIMA_HPP
#define BOOST_MATH_TOOLS_MINIMA_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <utility>
#include <boost/config/no_tr1/cmath.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/cstdint.hpp>

namespace boost{ namespace math{ namespace tools{

template <class F, class T>
std::pair<T, T> brent_find_minima(F f, T min, T max, int bits, boost::uintmax_t& max_iter)
BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(T) && noexcept(std::declval<F>()(std::declval<T>())))
{
BOOST_MATH_STD_USING
bits = (std::min)(policies::digits<T, policies::policy<> >() / 2, bits);
T tolerance = static_cast<T>(ldexp(1.0, 1-bits));
T x;  
T w;  
T v;  
T u;  
T delta;  
T delta2; 
T fu, fv, fw, fx;  
T mid; 
T fract1, fract2;  

static const T golden = 0.3819660f;  

x = w = v = max;
fw = fv = fx = f(x);
delta2 = delta = 0;

uintmax_t count = max_iter;

do{
mid = (min + max) / 2;
fract1 = tolerance * fabs(x) + tolerance / 4;
fract2 = 2 * fract1;
if(fabs(x - mid) <= (fract2 - (max - min) / 2))
break;

if(fabs(delta2) > fract1)
{
T r = (x - w) * (fx - fv);
T q = (x - v) * (fx - fw);
T p = (x - v) * q - (x - w) * r;
q = 2 * (q - r);
if(q > 0)
p = -p;
q = fabs(q);
T td = delta2;
delta2 = delta;
if((fabs(p) >= fabs(q * td / 2)) || (p <= q * (min - x)) || (p >= q * (max - x)))
{
delta2 = (x >= mid) ? min - x : max - x;
delta = golden * delta2;
}
else
{
delta = p / q;
u = x + delta;
if(((u - min) < fract2) || ((max- u) < fract2))
delta = (mid - x) < 0 ? (T)-fabs(fract1) : (T)fabs(fract1);
}
}
else
{
delta2 = (x >= mid) ? min - x : max - x;
delta = golden * delta2;
}
u = (fabs(delta) >= fract1) ? T(x + delta) : (delta > 0 ? T(x + fabs(fract1)) : T(x - fabs(fract1)));
fu = f(u);
if(fu <= fx)
{
if(u >= x)
min = x;
else
max = x;
v = w;
w = x;
x = u;
fv = fw;
fw = fx;
fx = fu;
}
else
{
if(u < x)
min = u;
else
max = u;
if((fu <= fw) || (w == x))
{
v = w;
w = u;
fv = fw;
fw = fu;
}
else if((fu <= fv) || (v == x) || (v == w))
{
v = u;
fv = fu;
}
}

}while(--count);

max_iter -= count;

return std::make_pair(x, fx);
}

template <class F, class T>
inline std::pair<T, T> brent_find_minima(F f, T min, T max, int digits)
BOOST_NOEXCEPT_IF(BOOST_MATH_IS_FLOAT(T) && noexcept(std::declval<F>()(std::declval<T>())))
{
boost::uintmax_t m = (std::numeric_limits<boost::uintmax_t>::max)();
return brent_find_minima(f, min, max, digits, m);
}

}}} 

#endif




