
#ifndef BOOST_MATH_TOOLS_SOLVE_ROOT_HPP
#define BOOST_MATH_TOOLS_SOLVE_ROOT_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/precision.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/cstdint.hpp>
#include <limits>

#ifdef BOOST_MATH_LOG_ROOT_ITERATIONS
#  define BOOST_MATH_LOGGER_INCLUDE <boost/math/tools/iteration_logger.hpp>
#  include BOOST_MATH_LOGGER_INCLUDE
#  undef BOOST_MATH_LOGGER_INCLUDE
#else
#  define BOOST_MATH_LOG_COUNT(count)
#endif

namespace boost{ namespace math{ namespace tools{

template <class T>
class eps_tolerance
{
public:
eps_tolerance()
{
eps = 4 * tools::epsilon<T>();
}
eps_tolerance(unsigned bits)
{
BOOST_MATH_STD_USING
eps = (std::max)(T(ldexp(1.0F, 1-bits)), T(4 * tools::epsilon<T>()));
}
bool operator()(const T& a, const T& b)
{
BOOST_MATH_STD_USING
return fabs(a - b) <= (eps * (std::min)(fabs(a), fabs(b)));
}
private:
T eps;
};

struct equal_floor
{
equal_floor(){}
template <class T>
bool operator()(const T& a, const T& b)
{
BOOST_MATH_STD_USING
return floor(a) == floor(b);
}
};

struct equal_ceil
{
equal_ceil(){}
template <class T>
bool operator()(const T& a, const T& b)
{
BOOST_MATH_STD_USING
return ceil(a) == ceil(b);
}
};

struct equal_nearest_integer
{
equal_nearest_integer(){}
template <class T>
bool operator()(const T& a, const T& b)
{
BOOST_MATH_STD_USING
return floor(a + 0.5f) == floor(b + 0.5f);
}
};

namespace detail{

template <class F, class T>
void bracket(F f, T& a, T& b, T c, T& fa, T& fb, T& d, T& fd)
{
BOOST_MATH_STD_USING  
T tol = tools::epsilon<T>() * 2;
if((b - a) < 2 * tol * a)
{
c = a + (b - a) / 2;
}
else if(c <= a + fabs(a) * tol)
{
c = a + fabs(a) * tol;
}
else if(c >= b - fabs(b) * tol)
{
c = b - fabs(b) * tol;
}
T fc = f(c);
if(fc == 0)
{
a = c;
fa = 0;
d = 0;
fd = 0;
return;
}
if(boost::math::sign(fa) * boost::math::sign(fc) < 0)
{
d = b;
fd = fb;
b = c;
fb = fc;
}
else
{
d = a;
fd = fa;
a = c;
fa= fc;
}
}

template <class T>
inline T safe_div(T num, T denom, T r)
{
BOOST_MATH_STD_USING  

if(fabs(denom) < 1)
{
if(fabs(denom * tools::max_value<T>()) <= fabs(num))
return r;
}
return num / denom;
}

template <class T>
inline T secant_interpolate(const T& a, const T& b, const T& fa, const T& fb)
{
BOOST_MATH_STD_USING  

T tol = tools::epsilon<T>() * 5;
T c = a - (fa / (fb - fa)) * (b - a);
if((c <= a + fabs(a) * tol) || (c >= b - fabs(b) * tol))
return (a + b) / 2;
return c;
}

template <class T>
T quadratic_interpolate(const T& a, const T& b, T const& d,
const T& fa, const T& fb, T const& fd, 
unsigned count)
{
T B = safe_div(T(fb - fa), T(b - a), tools::max_value<T>());
T A = safe_div(T(fd - fb), T(d - b), tools::max_value<T>());
A = safe_div(T(A - B), T(d - a), T(0));

if(A == 0)
{
return secant_interpolate(a, b, fa, fb);
}
T c;
if(boost::math::sign(A) * boost::math::sign(fa) > 0)
{
c = a;
}
else
{
c = b;
}
for(unsigned i = 1; i <= count; ++i)
{
c -= safe_div(T(fa+(B+A*(c-b))*(c-a)), T(B + A * (2 * c - a - b)), T(1 + c - a));
}
if((c <= a) || (c >= b))
{
c = secant_interpolate(a, b, fa, fb);
}
return c;
}

template <class T>
T cubic_interpolate(const T& a, const T& b, const T& d, 
const T& e, const T& fa, const T& fb, 
const T& fd, const T& fe)
{
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b
<< " d = " << d << " e = " << e << " fa = " << fa << " fb = " << fb 
<< " fd = " << fd << " fe = " << fe);
T q11 = (d - e) * fd / (fe - fd);
T q21 = (b - d) * fb / (fd - fb);
T q31 = (a - b) * fa / (fb - fa);
T d21 = (b - d) * fd / (fd - fb);
T d31 = (a - b) * fb / (fb - fa);
BOOST_MATH_INSTRUMENT_CODE(
"q11 = " << q11 << " q21 = " << q21 << " q31 = " << q31
<< " d21 = " << d21 << " d31 = " << d31);
T q22 = (d21 - q11) * fb / (fe - fb);
T q32 = (d31 - q21) * fa / (fd - fa);
T d32 = (d31 - q21) * fd / (fd - fa);
T q33 = (d32 - q22) * fa / (fe - fa);
T c = q31 + q32 + q33 + a;
BOOST_MATH_INSTRUMENT_CODE(
"q22 = " << q22 << " q32 = " << q32 << " d32 = " << d32
<< " q33 = " << q33 << " c = " << c);

if((c <= a) || (c >= b))
{
c = quadratic_interpolate(a, b, d, fa, fb, fd, 3);
BOOST_MATH_INSTRUMENT_CODE(
"Out of bounds interpolation, falling back to quadratic interpolation. c = " << c);
}

return c;
}

} 

template <class F, class T, class Tol, class Policy>
std::pair<T, T> toms748_solve(F f, const T& ax, const T& bx, const T& fax, const T& fbx, Tol tol, boost::uintmax_t& max_iter, const Policy& pol)
{
BOOST_MATH_STD_USING  

static const char* function = "boost::math::tools::toms748_solve<%1%>";

if (max_iter == 0)
return std::make_pair(ax, bx);

boost::uintmax_t count = max_iter;
T a, b, fa, fb, c, u, fu, a0, b0, d, fd, e, fe;
static const T mu = 0.5f;

a = ax;
b = bx;
if(a >= b)
return boost::math::detail::pair_from_single(policies::raise_domain_error(
function, 
"Parameters a and b out of order: a=%1%", a, pol));
fa = fax;
fb = fbx;

if(tol(a, b) || (fa == 0) || (fb == 0))
{
max_iter = 0;
if(fa == 0)
b = a;
else if(fb == 0)
a = b;
return std::make_pair(a, b);
}

if(boost::math::sign(fa) * boost::math::sign(fb) > 0)
return boost::math::detail::pair_from_single(policies::raise_domain_error(
function, 
"Parameters a and b do not bracket the root: a=%1%", a, pol));
fe = e = fd = 1e5F;

if(fa != 0)
{
c = detail::secant_interpolate(a, b, fa, fb);
detail::bracket(f, a, b, c, fa, fb, d, fd);
--count;
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b);

if(count && (fa != 0) && !tol(a, b))
{
c = detail::quadratic_interpolate(a, b, d, fa, fb, fd, 2);
e = d;
fe = fd;
detail::bracket(f, a, b, c, fa, fb, d, fd);
--count;
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b);
}
}

while(count && (fa != 0) && !tol(a, b))
{
a0 = a;
b0 = b;
T min_diff = tools::min_value<T>() * 32;
bool prof = (fabs(fa - fb) < min_diff) || (fabs(fa - fd) < min_diff) || (fabs(fa - fe) < min_diff) || (fabs(fb - fd) < min_diff) || (fabs(fb - fe) < min_diff) || (fabs(fd - fe) < min_diff);
if(prof)
{
c = detail::quadratic_interpolate(a, b, d, fa, fb, fd, 2);
BOOST_MATH_INSTRUMENT_CODE("Can't take cubic step!!!!");
}
else
{
c = detail::cubic_interpolate(a, b, d, e, fa, fb, fd, fe);
}
e = d;
fe = fd;
detail::bracket(f, a, b, c, fa, fb, d, fd);
if((0 == --count) || (fa == 0) || tol(a, b))
break;
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b);
prof = (fabs(fa - fb) < min_diff) || (fabs(fa - fd) < min_diff) || (fabs(fa - fe) < min_diff) || (fabs(fb - fd) < min_diff) || (fabs(fb - fe) < min_diff) || (fabs(fd - fe) < min_diff);
if(prof)
{
c = detail::quadratic_interpolate(a, b, d, fa, fb, fd, 3);
BOOST_MATH_INSTRUMENT_CODE("Can't take cubic step!!!!");
}
else
{
c = detail::cubic_interpolate(a, b, d, e, fa, fb, fd, fe);
}
detail::bracket(f, a, b, c, fa, fb, d, fd);
if((0 == --count) || (fa == 0) || tol(a, b))
break;
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b);
if(fabs(fa) < fabs(fb))
{
u = a;
fu = fa;
}
else
{
u = b;
fu = fb;
}
c = u - 2 * (fu / (fb - fa)) * (b - a);
if(fabs(c - u) > (b - a) / 2)
{
c = a + (b - a) / 2;
}
e = d;
fe = fd;
detail::bracket(f, a, b, c, fa, fb, d, fd);
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b);
BOOST_MATH_INSTRUMENT_CODE(" tol = " << T((fabs(a) - fabs(b)) / fabs(a)));
if((0 == --count) || (fa == 0) || tol(a, b))
break;
if((b - a) < mu * (b0 - a0))
continue;
e = d;
fe = fd;
detail::bracket(f, a, b, T(a + (b - a) / 2), fa, fb, d, fd);
--count;
BOOST_MATH_INSTRUMENT_CODE("Not converging: Taking a bisection!!!!");
BOOST_MATH_INSTRUMENT_CODE(" a = " << a << " b = " << b);
} 

max_iter -= count;
if(fa == 0)
{
b = a;
}
else if(fb == 0)
{
a = b;
}
BOOST_MATH_LOG_COUNT(max_iter)
return std::make_pair(a, b);
}

template <class F, class T, class Tol>
inline std::pair<T, T> toms748_solve(F f, const T& ax, const T& bx, const T& fax, const T& fbx, Tol tol, boost::uintmax_t& max_iter)
{
return toms748_solve(f, ax, bx, fax, fbx, tol, max_iter, policies::policy<>());
}

template <class F, class T, class Tol, class Policy>
inline std::pair<T, T> toms748_solve(F f, const T& ax, const T& bx, Tol tol, boost::uintmax_t& max_iter, const Policy& pol)
{
if (max_iter <= 2)
return std::make_pair(ax, bx);
max_iter -= 2;
std::pair<T, T> r = toms748_solve(f, ax, bx, f(ax), f(bx), tol, max_iter, pol);
max_iter += 2;
return r;
}

template <class F, class T, class Tol>
inline std::pair<T, T> toms748_solve(F f, const T& ax, const T& bx, Tol tol, boost::uintmax_t& max_iter)
{
return toms748_solve(f, ax, bx, tol, max_iter, policies::policy<>());
}

template <class F, class T, class Tol, class Policy>
std::pair<T, T> bracket_and_solve_root(F f, const T& guess, T factor, bool rising, Tol tol, boost::uintmax_t& max_iter, const Policy& pol)
{
BOOST_MATH_STD_USING
static const char* function = "boost::math::tools::bracket_and_solve_root<%1%>";
T a = guess;
T b = a;
T fa = f(a);
T fb = fa;
boost::uintmax_t count = max_iter - 1;

int step = 32;

if((fa < 0) == (guess < 0 ? !rising : rising))
{
while((boost::math::sign)(fb) == (boost::math::sign)(fa))
{
if(count == 0)
return boost::math::detail::pair_from_single(policies::raise_evaluation_error(function, "Unable to bracket root, last nearest value was %1%", b, pol));
if((max_iter - count) % step == 0)
{
factor *= 2;
if(step > 1) step /= 2;
}
a = b;
fa = fb;
b *= factor;
fb = f(b);
--count;
BOOST_MATH_INSTRUMENT_CODE("a = " << a << " b = " << b << " fa = " << fa << " fb = " << fb << " count = " << count);
}
}
else
{
while((boost::math::sign)(fb) == (boost::math::sign)(fa))
{
if(fabs(a) < tools::min_value<T>())
{
max_iter -= count;
max_iter += 1;
return a > 0 ? std::make_pair(T(0), T(a)) : std::make_pair(T(a), T(0)); 
}
if(count == 0)
return boost::math::detail::pair_from_single(policies::raise_evaluation_error(function, "Unable to bracket root, last nearest value was %1%", a, pol));
if((max_iter - count) % step == 0)
{
factor *= 2;
if(step > 1) step /= 2;
}
b = a;
fb = fa;
a /= factor;
fa = f(a);
--count;
BOOST_MATH_INSTRUMENT_CODE("a = " << a << " b = " << b << " fa = " << fa << " fb = " << fb << " count = " << count);
}
}
max_iter -= count;
max_iter += 1;
std::pair<T, T> r = toms748_solve(
f, 
(a < 0 ? b : a), 
(a < 0 ? a : b), 
(a < 0 ? fb : fa), 
(a < 0 ? fa : fb), 
tol, 
count, 
pol);
max_iter += count;
BOOST_MATH_INSTRUMENT_CODE("max_iter = " << max_iter << " count = " << count);
BOOST_MATH_LOG_COUNT(max_iter)
return r;
}

template <class F, class T, class Tol>
inline std::pair<T, T> bracket_and_solve_root(F f, const T& guess, const T& factor, bool rising, Tol tol, boost::uintmax_t& max_iter)
{
return bracket_and_solve_root(f, guess, factor, rising, tol, max_iter, policies::policy<>());
}

} 
} 
} 


#endif 

