
#ifndef BOOST_MATH_SPECIAL_FUNCTIONS_IBETA_INVERSE_HPP
#define BOOST_MATH_SPECIAL_FUNCTIONS_IBETA_INVERSE_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/math/special_functions/detail/t_distribution_inv.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T>
struct temme_root_finder
{
temme_root_finder(const T t_, const T a_) : t(t_), a(a_) {}

boost::math::tuple<T, T> operator()(T x)
{
BOOST_MATH_STD_USING 

T y = 1 - x;
if(y == 0)
{
T big = tools::max_value<T>() / 4;
return boost::math::make_tuple(static_cast<T>(-big), static_cast<T>(-big));
}
if(x == 0)
{
T big = tools::max_value<T>() / 4;
return boost::math::make_tuple(static_cast<T>(-big), big);
}
T f = log(x) + a * log(y) + t;
T f1 = (1 / x) - (a / (y));
return boost::math::make_tuple(f, f1);
}
private:
T t, a;
};
template <class T, class Policy>
T temme_method_1_ibeta_inverse(T a, T b, T z, const Policy& pol)
{
BOOST_MATH_STD_USING 

const T r2 = sqrt(T(2));
T eta0 = boost::math::erfc_inv(2 * z, pol);
eta0 /= -sqrt(a / 2);

T terms[4] = { eta0 };
T workspace[7];
T B = b - a;
T B_2 = B * B;
T B_3 = B_2 * B;

workspace[0] = -B * r2 / 2;
workspace[1] = (1 - 2 * B) / 8;
workspace[2] = -(B * r2 / 48);
workspace[3] = T(-1) / 192;
workspace[4] = -B * r2 / 3840;
terms[1] = tools::evaluate_polynomial(workspace, eta0, 5);
workspace[0] = B * r2 * (3 * B - 2) / 12;
workspace[1] = (20 * B_2 - 12 * B + 1) / 128;
workspace[2] = B * r2 * (20 * B - 1) / 960;
workspace[3] = (16 * B_2 + 30 * B - 15) / 4608;
workspace[4] = B * r2 * (21 * B + 32) / 53760;
workspace[5] = (-32 * B_2 + 63) / 368640;
workspace[6] = -B * r2 * (120 * B + 17) / 25804480;
terms[2] = tools::evaluate_polynomial(workspace, eta0, 7);
workspace[0] = B * r2 * (-75 * B_2 + 80 * B - 16) / 480;
workspace[1] = (-1080 * B_3 + 868 * B_2 - 90 * B - 45) / 9216;
workspace[2] = B * r2 * (-1190 * B_2 + 84 * B + 373) / 53760;
workspace[3] = (-2240 * B_3 - 2508 * B_2 + 2100 * B - 165) / 368640;
terms[3] = tools::evaluate_polynomial(workspace, eta0, 4);
T eta = tools::evaluate_polynomial(terms, T(1/a), 4);
T eta_2 = eta * eta;
T c = -exp(-eta_2 / 2);
T x;
if(eta_2 == 0)
x = 0.5;
else
x = (1 + eta * sqrt((1 + c) / eta_2)) / 2;

BOOST_ASSERT(x >= 0);
BOOST_ASSERT(x <= 1);
BOOST_ASSERT(eta * (x - 0.5) >= 0);
#ifdef BOOST_INSTRUMENT
std::cout << "Estimating x with Temme method 1: " << x << std::endl;
#endif
return x;
}
template <class T, class Policy>
T temme_method_2_ibeta_inverse(T , T , T z, T r, T theta, const Policy& pol)
{
BOOST_MATH_STD_USING 

T eta0 = boost::math::erfc_inv(2 * z, pol);
eta0 /= -sqrt(r / 2);

T s = sin(theta);
T c = cos(theta);
T terms[4] = { eta0 };
T workspace[6];
T sc = s * c;
T sc_2 = sc * sc;
T sc_3 = sc_2 * sc;
T sc_4 = sc_2 * sc_2;
T sc_5 = sc_2 * sc_3;
T sc_6 = sc_3 * sc_3;
T sc_7 = sc_4 * sc_3;
workspace[0] = (2 * s * s - 1) / (3 * s * c);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co1[] = { -1, -5, 5 };
workspace[1] = -tools::evaluate_even_polynomial(co1, s, 3) / (36 * sc_2);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co2[] = { 1, 21, -69, 46 };
workspace[2] = tools::evaluate_even_polynomial(co2, s, 4) / (1620 * sc_3);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co3[] = { 7, -2, 33, -62, 31 };
workspace[3] = -tools::evaluate_even_polynomial(co3, s, 5) / (6480 * sc_4);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co4[] = { 25, -52, -17, 88, -115, 46 };
workspace[4] = tools::evaluate_even_polynomial(co4, s, 6) / (90720 * sc_5);
terms[1] = tools::evaluate_polynomial(workspace, eta0, 5);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co5[] = { 7, 12, -78, 52 };
workspace[0] = -tools::evaluate_even_polynomial(co5, s, 4) / (405 * sc_3);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co6[] = { -7, 2, 183, -370, 185 };
workspace[1] = tools::evaluate_even_polynomial(co6, s, 5) / (2592 * sc_4);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co7[] = { -533, 776, -1835, 10240, -13525, 5410 };
workspace[2] = -tools::evaluate_even_polynomial(co7, s, 6) / (204120 * sc_5);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co8[] = { -1579, 3747, -3372, -15821, 45588, -45213, 15071 };
workspace[3] = -tools::evaluate_even_polynomial(co8, s, 7) / (2099520 * sc_6);
terms[2] = tools::evaluate_polynomial(workspace, eta0, 4);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co9[] = {449, -1259, -769, 6686, -9260, 3704 };
workspace[0] = tools::evaluate_even_polynomial(co9, s, 6) / (102060 * sc_5);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co10[] = { 63149, -151557, 140052, -727469, 2239932, -2251437, 750479 };
workspace[1] = -tools::evaluate_even_polynomial(co10, s, 7) / (20995200 * sc_6);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co11[] = { 29233, -78755, 105222, 146879, -1602610, 3195183, -2554139, 729754 };
workspace[2] = tools::evaluate_even_polynomial(co11, s, 8) / (36741600 * sc_7);
terms[3] = tools::evaluate_polynomial(workspace, eta0, 3);
T eta = tools::evaluate_polynomial(terms, T(1/r), 4);
T x;
T s_2 = s * s;
T c_2 = c * c;
T alpha = c / s;
alpha *= alpha;
T lu = (-(eta * eta) / (2 * s_2) + log(s_2) + c_2 * log(c_2) / s_2);
if(fabs(eta) < 0.7)
{
workspace[0] = s * s;
workspace[1] = s * c;
workspace[2] = (1 - 2 * workspace[0]) / 3;
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co12[] = { 1, -13, 13 };
workspace[3] = tools::evaluate_polynomial(co12, workspace[0], 3) / (36 * s * c);
static const BOOST_MATH_INT_TABLE_TYPE(T, int) co13[] = { 1, 21, -69, 46 };
workspace[4] = tools::evaluate_polynomial(co13, workspace[0], 4) / (270 * workspace[0] * c * c);
x = tools::evaluate_polynomial(workspace, eta, 5);
#ifdef BOOST_INSTRUMENT
std::cout << "Estimating x with Temme method 2 (small eta): " << x << std::endl;
#endif
}
else
{
T u = exp(lu);
workspace[0] = u;
workspace[1] = alpha;
workspace[2] = 0;
workspace[3] = 3 * alpha * (3 * alpha + 1) / 6;
workspace[4] = 4 * alpha * (4 * alpha + 1) * (4 * alpha + 2) / 24;
workspace[5] = 5 * alpha * (5 * alpha + 1) * (5 * alpha + 2) * (5 * alpha + 3) / 120;
x = tools::evaluate_polynomial(workspace, u, 6);
if((x - s_2) * eta < 0)
x = 1 - x;
#ifdef BOOST_INSTRUMENT
std::cout << "Estimating x with Temme method 2 (large eta): " << x << std::endl;
#endif
}
T lower, upper;
if(eta < 0)
{
lower = 0;
upper = s_2;
}
else
{
lower = s_2;
upper = 1;
}
if((x < lower) || (x > upper))
x = (lower+upper) / 2;
x = tools::newton_raphson_iterate(
temme_root_finder<T>(-lu, alpha), x, lower, upper, policies::digits<T, Policy>() / 2);

return x;
}
template <class T, class Policy>
T temme_method_3_ibeta_inverse(T a, T b, T p, T q, const Policy& pol)
{
BOOST_MATH_STD_USING 

T eta0;
if(p < q)
eta0 = boost::math::gamma_q_inv(b, p, pol);
else
eta0 = boost::math::gamma_p_inv(b, q, pol);
eta0 /= a;
T mu = b / a;
T w = sqrt(1 + mu);
T w_2 = w * w;
T w_3 = w_2 * w;
T w_4 = w_2 * w_2;
T w_5 = w_3 * w_2;
T w_6 = w_3 * w_3;
T w_7 = w_4 * w_3;
T w_8 = w_4 * w_4;
T w_9 = w_5 * w_4;
T w_10 = w_5 * w_5;
T d = eta0 - mu;
T d_2 = d * d;
T d_3 = d_2 * d;
T d_4 = d_2 * d_2;
T w1 = w + 1;
T w1_2 = w1 * w1;
T w1_3 = w1 * w1_2;
T w1_4 = w1_2 * w1_2;
T e1 = (w + 2) * (w - 1) / (3 * w);
e1 += (w_3 + 9 * w_2 + 21 * w + 5) * d / (36 * w_2 * w1);
e1 -= (w_4 - 13 * w_3 + 69 * w_2 + 167 * w + 46) * d_2 / (1620 * w1_2 * w_3);
e1 -= (7 * w_5 + 21 * w_4 + 70 * w_3 + 26 * w_2 - 93 * w - 31) * d_3 / (6480 * w1_3 * w_4);
e1 -= (75 * w_6 + 202 * w_5 + 188 * w_4 - 888 * w_3 - 1345 * w_2 + 118 * w + 138) * d_4 / (272160 * w1_4 * w_5);

T e2 = (28 * w_4 + 131 * w_3 + 402 * w_2 + 581 * w + 208) * (w - 1) / (1620 * w1 * w_3);
e2 -= (35 * w_6 - 154 * w_5 - 623 * w_4 - 1636 * w_3 - 3983 * w_2 - 3514 * w - 925) * d / (12960 * w1_2 * w_4);
e2 -= (2132 * w_7 + 7915 * w_6 + 16821 * w_5 + 35066 * w_4 + 87490 * w_3 + 141183 * w_2 + 95993 * w + 21640) * d_2  / (816480 * w_5 * w1_3);
e2 -= (11053 * w_8 + 53308 * w_7 + 117010 * w_6 + 163924 * w_5 + 116188 * w_4 - 258428 * w_3 - 677042 * w_2 - 481940 * w - 105497) * d_3 / (14696640 * w1_4 * w_6);

T e3 = -((3592 * w_7 + 8375 * w_6 - 1323 * w_5 - 29198 * w_4 - 89578 * w_3 - 154413 * w_2 - 116063 * w - 29632) * (w - 1)) / (816480 * w_5 * w1_2);
e3 -= (442043 * w_9 + 2054169 * w_8 + 3803094 * w_7 + 3470754 * w_6 + 2141568 * w_5 - 2393568 * w_4 - 19904934 * w_3 - 34714674 * w_2 - 23128299 * w - 5253353) * d / (146966400 * w_6 * w1_3);
e3 -= (116932 * w_10 + 819281 * w_9 + 2378172 * w_8 + 4341330 * w_7 + 6806004 * w_6 + 10622748 * w_5 + 18739500 * w_4 + 30651894 * w_3 + 30869976 * w_2 + 15431867 * w + 2919016) * d_2 / (146966400 * w1_4 * w_7);
T eta = eta0 + e1 / a + e2 / (a * a) + e3 / (a * a * a);
if(eta <= 0)
eta = tools::min_value<T>();
T u = eta - mu * log(eta) + (1 + mu) * log(1 + mu) - mu;
T cross = 1 / (1 + mu);
T lower = eta < mu ? cross : 0;
T upper = eta < mu ? 1 : cross;
T x = (lower + upper) / 2;
x = tools::newton_raphson_iterate(
temme_root_finder<T>(u, mu), x, lower, upper, policies::digits<T, Policy>() / 2);
#ifdef BOOST_INSTRUMENT
std::cout << "Estimating x with Temme method 3: " << x << std::endl;
#endif
return x;
}

template <class T, class Policy>
struct ibeta_roots
{
ibeta_roots(T _a, T _b, T t, bool inv = false)
: a(_a), b(_b), target(t), invert(inv) {}

boost::math::tuple<T, T, T> operator()(T x)
{
BOOST_MATH_STD_USING 

BOOST_FPU_EXCEPTION_GUARD

T f1;
T y = 1 - x;
T f = ibeta_imp(a, b, x, Policy(), invert, true, &f1) - target;
if(invert)
f1 = -f1;
if(y == 0)
y = tools::min_value<T>() * 64;
if(x == 0)
x = tools::min_value<T>() * 64;

T f2 = f1 * (-y * a + (b - 2) * x + 1);
if(fabs(f2) < y * x * tools::max_value<T>())
f2 /= (y * x);
if(invert)
f2 = -f2;

if(f1 == 0)
f1 = (invert ? -1 : 1) * tools::min_value<T>() * 64;

return boost::math::make_tuple(f, f1, f2);
}
private:
T a, b, target;
bool invert;
};

template <class T, class Policy>
T ibeta_inv_imp(T a, T b, T p, T q, const Policy& pol, T* py)
{
BOOST_MATH_STD_USING  

bool invert = false;
if(q == 0)
{
if(py) *py = 0;
return 1;
}
else if(p == 0)
{
if(py) *py = 1;
return 0;
}
else if(a == 1)
{
if(b == 1)
{
if(py) *py = 1 - p;
return p;
}
std::swap(a, b);
std::swap(p, q);
invert = true;
}
T x = 0; 
T y; 

T lower = 0;
T upper = 1;
if(a == 0.5f)
{
if(b == 0.5f)
{
x = sin(p * constants::half_pi<T>());
x *= x;
if(py)
{
*py = sin(q * constants::half_pi<T>());
*py *= *py;
}
return x;
}
else if(b > 0.5f)
{
std::swap(a, b);
std::swap(p, q);
invert = !invert;
}
}
if((b == 0.5f) && (a >= 0.5f) && (p != 1))
{
x = find_ibeta_inv_from_t_dist(a, p, q, &y, pol);
}
else if(b == 1)
{
if(p < q)
{
if(a > 1)
{
x = pow(p, 1 / a);
y = -boost::math::expm1(log(p) / a, pol);
}
else
{
x = pow(p, 1 / a);
y = 1 - x;
}
}
else
{
x = exp(boost::math::log1p(-q, pol) / a);
y = -boost::math::expm1(boost::math::log1p(-q, pol) / a, pol);
}
if(invert)
std::swap(x, y);
if(py)
*py = y;
return x;
}
else if(a + b > 5)
{
if(p > 0.5)
{
std::swap(a, b);
std::swap(p, q);
invert = !invert;
}
T minv = (std::min)(a, b);
T maxv = (std::max)(a, b);
if((sqrt(minv) > (maxv - minv)) && (minv > 5))
{
x = temme_method_1_ibeta_inverse(a, b, p, pol);
y = 1 - x;
}
else
{
T r = a + b;
T theta = asin(sqrt(a / r));
T lambda = minv / r;
if((lambda >= 0.2) && (lambda <= 0.8) && (r >= 10))
{
T ppa = pow(p, 1/a);
if((ppa < 0.0025) && (a + b < 200))
{
x = ppa * pow(a * boost::math::beta(a, b, pol), 1/a);
}
else
x = temme_method_2_ibeta_inverse(a, b, p, r, theta, pol);
y = 1 - x;
}
else
{
if(a < b)
{
std::swap(a, b);
std::swap(p, q);
invert = !invert;
}
T bet = 0;
if(b < 2)
bet = boost::math::beta(a, b, pol);
if(bet != 0)
{
y = pow(b * q * bet, 1/b);
x = 1 - y;
}
else 
y = 1;
if(y > 1e-5)
{
x = temme_method_3_ibeta_inverse(a, b, p, q, pol);
y = 1 - x;
}
}
}
}
else if((a < 1) && (b < 1))
{
T xs = (1 - a) / (2 - a - b);
T fs = boost::math::ibeta(a, b, xs, pol) - p;
if(fabs(fs) / p < tools::epsilon<T>() * 3)
{
*py = invert ? xs : 1 - xs;
return invert ? 1-xs : xs;
}
if(fs < 0)
{
std::swap(a, b);
std::swap(p, q);
invert = !invert;
xs = 1 - xs;
}
if (a < tools::min_value<T>())
{
if (py)
{
*py = invert ? 0 : 1;
}
return invert ? 1 : 0; 
}
T bet = 0;
T xg;
bool overflow = false;
try {
bet = boost::math::beta(a, b, pol);
}
catch (const std::runtime_error&)
{
overflow = true;
}
if (overflow || !(boost::math::isfinite)(bet))
{
xg = exp((boost::math::lgamma(a + 1, pol) + boost::math::lgamma(b, pol) - boost::math::lgamma(a + b, pol) + log(p)) / a);
}
else
xg = pow(a * p * bet, 1/a);
x = xg / (1 + xg);
y = 1 / (1 + xg);
if(x > xs)
x = xs;
upper = xs;
}
else if((a > 1) && (b > 1))
{
T xs = (a - 1) / (a + b - 2);
T xs2 = (b - 1) / (a + b - 2);
T ps = boost::math::ibeta(a, b, xs, pol) - p;

if(ps < 0)
{
std::swap(a, b);
std::swap(p, q);
std::swap(xs, xs2);
invert = !invert;
}
T lx = log(p * a * boost::math::beta(a, b, pol)) / a;
x = exp(lx);
y = x < 0.9 ? T(1 - x) : (T)(-boost::math::expm1(lx, pol));

if((b < a) && (x < 0.2))
{
T ap1 = a - 1;
T bm1 = b - 1;
T a_2 = a * a;
T a_3 = a * a_2;
T b_2 = b * b;
T terms[5] = { 0, 1 };
terms[2] = bm1 / ap1;
ap1 *= ap1;
terms[3] = bm1 * (3 * a * b + 5 * b + a_2 - a - 4) / (2 * (a + 2) * ap1);
ap1 *= (a + 1);
terms[4] = bm1 * (33 * a * b_2 + 31 * b_2 + 8 * a_2 * b_2 - 30 * a * b - 47 * b + 11 * a_2 * b + 6 * a_3 * b + 18 + 4 * a - a_3 + a_2 * a_2 - 10 * a_2)
/ (3 * (a + 3) * (a + 2) * ap1);
x = tools::evaluate_polynomial(terms, x, 5);
}
if(x > xs)
x = xs;
upper = xs;
}
else 
{
if(b < a)
{
std::swap(a, b);
std::swap(p, q);
invert = !invert;
}
if(pow(p, 1/a) < 0.5)
{
x = pow(p * a * boost::math::beta(a, b, pol), 1 / a);
if(x == 0)
x = boost::math::tools::min_value<T>();
y = 1 - x;
}
else 
{
y = pow(1 - pow(p, b * boost::math::beta(a, b, pol)), 1/b);
if(y == 0)
y = boost::math::tools::min_value<T>();
x = 1 - y;
}
}

if(x > 0.5)
{
std::swap(a, b);
std::swap(p, q);
std::swap(x, y);
invert = !invert;
T l = 1 - upper;
T u = 1 - lower;
lower = l;
upper = u;
}
if(lower == 0)
{
if(invert && (py == 0))
{
lower = boost::math::tools::epsilon<T>();
if(x < lower)
x = lower;
}
else
lower = boost::math::tools::min_value<T>();
if(x < lower)
x = lower;
}
int digits = boost::math::policies::digits<T, Policy>() / 2;
if((x < 1e-50) && ((a < 1) || (b < 1)))
{
digits *= 3;  
digits /= 2;
}
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
x = boost::math::tools::halley_iterate(
boost::math::detail::ibeta_roots<T, Policy>(a, b, (p < q ? p : q), (p < q ? false : true)), x, lower, upper, digits, max_iter);
policies::check_root_iterations<T>("boost::math::ibeta<%1%>(%1%, %1%, %1%)", max_iter, pol);
if(x == lower)
x = 0;
if(py)
*py = invert ? x : 1 - x;
return invert ? 1-x : x;
}

} 

template <class T1, class T2, class T3, class T4, class Policy>
inline typename tools::promote_args<T1, T2, T3, T4>::type  
ibeta_inv(T1 a, T2 b, T3 p, T4* py, const Policy& pol)
{
static const char* function = "boost::math::ibeta_inv<%1%>(%1%,%1%,%1%)";
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2, T3, T4>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

if(a <= 0)
return policies::raise_domain_error<result_type>(function, "The argument a to the incomplete beta function inverse must be greater than zero (got a=%1%).", a, pol);
if(b <= 0)
return policies::raise_domain_error<result_type>(function, "The argument b to the incomplete beta function inverse must be greater than zero (got b=%1%).", b, pol);
if((p < 0) || (p > 1))
return policies::raise_domain_error<result_type>(function, "Argument p outside the range [0,1] in the incomplete beta function inverse (got p=%1%).", p, pol);

value_type rx, ry;

rx = detail::ibeta_inv_imp(
static_cast<value_type>(a),
static_cast<value_type>(b),
static_cast<value_type>(p),
static_cast<value_type>(1 - p),
forwarding_policy(), &ry);

if(py) *py = policies::checked_narrowing_cast<T4, forwarding_policy>(ry, function);
return policies::checked_narrowing_cast<result_type, forwarding_policy>(rx, function);
}

template <class T1, class T2, class T3, class T4>
inline typename tools::promote_args<T1, T2, T3, T4>::type  
ibeta_inv(T1 a, T2 b, T3 p, T4* py)
{
return ibeta_inv(a, b, p, py, policies::policy<>());
}

template <class T1, class T2, class T3>
inline typename tools::promote_args<T1, T2, T3>::type 
ibeta_inv(T1 a, T2 b, T3 p)
{
typedef typename tools::promote_args<T1, T2, T3>::type result_type;
return ibeta_inv(a, b, p, static_cast<result_type*>(0), policies::policy<>());
}

template <class T1, class T2, class T3, class Policy>
inline typename tools::promote_args<T1, T2, T3>::type 
ibeta_inv(T1 a, T2 b, T3 p, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2, T3>::type result_type;
return ibeta_inv(a, b, p, static_cast<result_type*>(0), pol);
}

template <class T1, class T2, class T3, class T4, class Policy>
inline typename tools::promote_args<T1, T2, T3, T4>::type 
ibetac_inv(T1 a, T2 b, T3 q, T4* py, const Policy& pol)
{
static const char* function = "boost::math::ibetac_inv<%1%>(%1%,%1%,%1%)";
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2, T3, T4>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

if(a <= 0)
return policies::raise_domain_error<result_type>(function, "The argument a to the incomplete beta function inverse must be greater than zero (got a=%1%).", a, pol);
if(b <= 0)
return policies::raise_domain_error<result_type>(function, "The argument b to the incomplete beta function inverse must be greater than zero (got b=%1%).", b, pol);
if((q < 0) || (q > 1))
return policies::raise_domain_error<result_type>(function, "Argument q outside the range [0,1] in the incomplete beta function inverse (got q=%1%).", q, pol);

value_type rx, ry;

rx = detail::ibeta_inv_imp(
static_cast<value_type>(a),
static_cast<value_type>(b),
static_cast<value_type>(1 - q),
static_cast<value_type>(q),
forwarding_policy(), &ry);

if(py) *py = policies::checked_narrowing_cast<T4, forwarding_policy>(ry, function);
return policies::checked_narrowing_cast<result_type, forwarding_policy>(rx, function);
}

template <class T1, class T2, class T3, class T4>
inline typename tools::promote_args<T1, T2, T3, T4>::type 
ibetac_inv(T1 a, T2 b, T3 q, T4* py)
{
return ibetac_inv(a, b, q, py, policies::policy<>());
}

template <class RT1, class RT2, class RT3>
inline typename tools::promote_args<RT1, RT2, RT3>::type 
ibetac_inv(RT1 a, RT2 b, RT3 q)
{
typedef typename tools::promote_args<RT1, RT2, RT3>::type result_type;
return ibetac_inv(a, b, q, static_cast<result_type*>(0), policies::policy<>());
}

template <class RT1, class RT2, class RT3, class Policy>
inline typename tools::promote_args<RT1, RT2, RT3>::type
ibetac_inv(RT1 a, RT2 b, RT3 q, const Policy& pol)
{
typedef typename tools::promote_args<RT1, RT2, RT3>::type result_type;
return ibetac_inv(a, b, q, static_cast<result_type*>(0), pol);
}

} 
} 

#endif 




