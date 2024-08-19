

#ifndef BOOST_MATH_SF_GAMMA_HPP
#define BOOST_MATH_SF_GAMMA_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/math/tools/series.hpp>
#include <boost/math/tools/fraction.hpp>
#include <boost/math/tools/precision.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/special_functions/powm1.hpp>
#include <boost/math/special_functions/sqrt1pm1.hpp>
#include <boost/math/special_functions/lanczos.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/detail/igamma_large.hpp>
#include <boost/math/special_functions/detail/unchecked_factorial.hpp>
#include <boost/math/special_functions/detail/lgamma_small.hpp>
#include <boost/math/special_functions/bernoulli.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/assert.hpp>
#include <boost/mpl/greater.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/mpl/greater.hpp>

#include <boost/config/no_tr1/cmath.hpp>
#include <algorithm>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4702) 
# pragma warning(disable: 4127) 
# pragma warning(disable: 4100) 
#endif

namespace boost{ namespace math{

namespace detail{

template <class T>
inline bool is_odd(T v, const boost::true_type&)
{
int i = static_cast<int>(v);
return i&1;
}
template <class T>
inline bool is_odd(T v, const boost::false_type&)
{
BOOST_MATH_STD_USING
T modulus = v - 2 * floor(v/2);
return static_cast<bool>(modulus != 0);
}
template <class T>
inline bool is_odd(T v)
{
return is_odd(v, ::boost::is_convertible<T, int>());
}

template <class T>
T sinpx(T z)
{
BOOST_MATH_STD_USING
int sign = 1;
if(z < 0)
{
z = -z;
}
T fl = floor(z);
T dist;
if(is_odd(fl))
{
fl += 1;
dist = fl - z;
sign = -sign;
}
else
{
dist = z - fl;
}
BOOST_ASSERT(fl >= 0);
if(dist > 0.5)
dist = 1 - dist;
T result = sin(dist*boost::math::constants::pi<T>());
return sign*z*result;
} 
template <class T, class Policy, class Lanczos>
T gamma_imp(T z, const Policy& pol, const Lanczos& l)
{
BOOST_MATH_STD_USING

T result = 1;

#ifdef BOOST_MATH_INSTRUMENT
static bool b = false;
if(!b)
{
std::cout << "tgamma_imp called with " << typeid(z).name() << " " << typeid(l).name() << std::endl;
b = true;
}
#endif
static const char* function = "boost::math::tgamma<%1%>(%1%)";

if(z <= 0)
{
if(floor(z) == z)
return policies::raise_pole_error<T>(function, "Evaluation of tgamma at a negative integer %1%.", z, pol);
if(z <= -20)
{
result = gamma_imp(T(-z), pol, l) * sinpx(z);
BOOST_MATH_INSTRUMENT_VARIABLE(result);
if((fabs(result) < 1) && (tools::max_value<T>() * fabs(result) < boost::math::constants::pi<T>()))
return -boost::math::sign(result) * policies::raise_overflow_error<T>(function, "Result of tgamma is too large to represent.", pol);
result = -boost::math::constants::pi<T>() / result;
if(result == 0)
return policies::raise_underflow_error<T>(function, "Result of tgamma is too small to represent.", pol);
if((boost::math::fpclassify)(result) == (int)FP_SUBNORMAL)
return policies::raise_denorm_error<T>(function, "Result of tgamma is denormalized.", result, pol);
BOOST_MATH_INSTRUMENT_VARIABLE(result);
return result;
}

while(z < 0)
{
result /= z;
z += 1;
}
}
BOOST_MATH_INSTRUMENT_VARIABLE(result);
if((floor(z) == z) && (z < max_factorial<T>::value))
{
result *= unchecked_factorial<T>(itrunc(z, pol) - 1);
BOOST_MATH_INSTRUMENT_VARIABLE(result);
}
else if (z < tools::root_epsilon<T>())
{
if (z < 1 / tools::max_value<T>())
result = policies::raise_overflow_error<T>(function, 0, pol);
result *= 1 / z - constants::euler<T>();
}
else
{
result *= Lanczos::lanczos_sum(z);
T zgh = (z + static_cast<T>(Lanczos::g()) - boost::math::constants::half<T>());
T lzgh = log(zgh);
BOOST_MATH_INSTRUMENT_VARIABLE(result);
BOOST_MATH_INSTRUMENT_VARIABLE(tools::log_max_value<T>());
if(z * lzgh > tools::log_max_value<T>())
{
BOOST_MATH_INSTRUMENT_VARIABLE(zgh);
if(lzgh * z / 2 > tools::log_max_value<T>())
return boost::math::sign(result) * policies::raise_overflow_error<T>(function, "Result of tgamma is too large to represent.", pol);
T hp = pow(zgh, (z / 2) - T(0.25));
BOOST_MATH_INSTRUMENT_VARIABLE(hp);
result *= hp / exp(zgh);
BOOST_MATH_INSTRUMENT_VARIABLE(result);
if(tools::max_value<T>() / hp < result)
return boost::math::sign(result) * policies::raise_overflow_error<T>(function, "Result of tgamma is too large to represent.", pol);
result *= hp;
BOOST_MATH_INSTRUMENT_VARIABLE(result);
}
else
{
BOOST_MATH_INSTRUMENT_VARIABLE(zgh);
BOOST_MATH_INSTRUMENT_VARIABLE(pow(zgh, z - boost::math::constants::half<T>()));
BOOST_MATH_INSTRUMENT_VARIABLE(exp(zgh));
result *= pow(zgh, z - boost::math::constants::half<T>()) / exp(zgh);
BOOST_MATH_INSTRUMENT_VARIABLE(result);
}
}
return result;
}
template <class T, class Policy, class Lanczos>
T lgamma_imp(T z, const Policy& pol, const Lanczos& l, int* sign = 0)
{
#ifdef BOOST_MATH_INSTRUMENT
static bool b = false;
if(!b)
{
std::cout << "lgamma_imp called with " << typeid(z).name() << " " << typeid(l).name() << std::endl;
b = true;
}
#endif

BOOST_MATH_STD_USING

static const char* function = "boost::math::lgamma<%1%>(%1%)";

T result = 0;
int sresult = 1;
if(z <= -tools::root_epsilon<T>())
{
if(floor(z) == z)
return policies::raise_pole_error<T>(function, "Evaluation of lgamma at a negative integer %1%.", z, pol);

T t = sinpx(z);
z = -z;
if(t < 0)
{
t = -t;
}
else
{
sresult = -sresult;
}
result = log(boost::math::constants::pi<T>()) - lgamma_imp(z, pol, l) - log(t);
}
else if (z < tools::root_epsilon<T>())
{
if (0 == z)
return policies::raise_pole_error<T>(function, "Evaluation of lgamma at %1%.", z, pol);
if (4 * fabs(z) < tools::epsilon<T>())
result = -log(fabs(z));
else
result = log(fabs(1 / z - constants::euler<T>()));
if (z < 0)
sresult = -1;
}
else if(z < 15)
{
typedef typename policies::precision<T, Policy>::type precision_type;
typedef boost::integral_constant<int,
precision_type::value <= 0 ? 0 :
precision_type::value <= 64 ? 64 :
precision_type::value <= 113 ? 113 : 0
> tag_type;

result = lgamma_small_imp<T>(z, T(z - 1), T(z - 2), tag_type(), pol, l);
}
else if((z >= 3) && (z < 100) && (std::numeric_limits<T>::max_exponent >= 1024))
{
result = log(gamma_imp(z, pol, l));
}
else
{
T zgh = static_cast<T>(z + Lanczos::g() - boost::math::constants::half<T>());
result = log(zgh) - 1;
result *= z - 0.5f;
if(result * tools::epsilon<T>() < 20)
result += log(Lanczos::lanczos_sum_expG_scaled(z));
}

if(sign)
*sign = sresult;
return result;
}

template <class T>
struct upper_incomplete_gamma_fract
{
private:
T z, a;
int k;
public:
typedef std::pair<T,T> result_type;

upper_incomplete_gamma_fract(T a1, T z1)
: z(z1-a1+1), a(a1), k(0)
{
}

result_type operator()()
{
++k;
z += 2;
return result_type(k * (a - k), z);
}
};

template <class T>
inline T upper_gamma_fraction(T a, T z, T eps)
{
upper_incomplete_gamma_fract<T> f(a, z);
return 1 / (z - a + 1 + boost::math::tools::continued_fraction_a(f, eps));
}

template <class T>
struct lower_incomplete_gamma_series
{
private:
T a, z, result;
public:
typedef T result_type;
lower_incomplete_gamma_series(T a1, T z1) : a(a1), z(z1), result(1){}

T operator()()
{
T r = result;
a += 1;
result *= z/a;
return r;
}
};

template <class T, class Policy>
inline T lower_gamma_series(T a, T z, const Policy& pol, T init_value = 0)
{
lower_incomplete_gamma_series<T> s(a, z);
boost::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();
T factor = policies::get_epsilon<T, Policy>();
T result = boost::math::tools::sum_series(s, factor, max_iter, init_value);
policies::check_series_iterations<T>("boost::math::detail::lower_gamma_series<%1%>(%1%)", max_iter, pol);
return result;
}

template<class T>
std::size_t highest_bernoulli_index()
{
const float digits10_of_type = (std::numeric_limits<T>::is_specialized
? static_cast<float>(std::numeric_limits<T>::digits10)
: static_cast<float>(boost::math::tools::digits<T>() * 0.301F));

return static_cast<std::size_t>(18.0F + (0.6F * digits10_of_type));
}

template<class T>
int minimum_argument_for_bernoulli_recursion()
{
const float digits10_of_type = (std::numeric_limits<T>::is_specialized
? static_cast<float>(std::numeric_limits<T>::digits10)
: static_cast<float>(boost::math::tools::digits<T>() * 0.301F));

const float limit = std::ceil(std::pow(1.0f / std::ldexp(1.0f, 1-boost::math::tools::digits<T>()), 1.0f / 20.0f));

return (int)((std::min)(digits10_of_type * 1.7F, limit));
}

template <class T, class Policy>
T scaled_tgamma_no_lanczos(const T& z, const Policy& pol, bool islog = false)
{
BOOST_MATH_STD_USING
BOOST_ASSERT(minimum_argument_for_bernoulli_recursion<T>() <= z);


const std::size_t number_of_bernoullis_b2n = policies::get_max_series_iterations<Policy>();

T one_over_x_pow_two_n_minus_one = 1 / z;
const T one_over_x2 = one_over_x_pow_two_n_minus_one * one_over_x_pow_two_n_minus_one;
T sum = (boost::math::bernoulli_b2n<T>(1) / 2) * one_over_x_pow_two_n_minus_one;
const T target_epsilon_to_break_loop = sum * boost::math::tools::epsilon<T>();
const T half_ln_two_pi_over_z = sqrt(boost::math::constants::two_pi<T>() / z);
T last_term = 2 * sum;

for (std::size_t n = 2U;; ++n)
{
one_over_x_pow_two_n_minus_one *= one_over_x2;

const std::size_t n2 = static_cast<std::size_t>(n * 2U);

const T term = (boost::math::bernoulli_b2n<T>(static_cast<int>(n)) * one_over_x_pow_two_n_minus_one) / (n2 * (n2 - 1U));

if ((n >= 3U) && (abs(term) < target_epsilon_to_break_loop))
{

break;
}
if (n > number_of_bernoullis_b2n)
return policies::raise_evaluation_error("scaled_tgamma_no_lanczos<%1%>()", "Exceeded maximum series iterations without reaching convergence, best approximation was %1%", T(exp(sum) * half_ln_two_pi_over_z), pol);

sum += term;

T fterm = fabs(term);
if(fterm > last_term)
return policies::raise_evaluation_error("scaled_tgamma_no_lanczos<%1%>()", "Series became divergent without reaching convergence, best approximation was %1%", T(exp(sum) * half_ln_two_pi_over_z), pol);
last_term = fterm;
}

T scaled_gamma_value = islog ? T(sum + log(half_ln_two_pi_over_z)) : T(exp(sum) * half_ln_two_pi_over_z);
return scaled_gamma_value;
}

template <class T, class Policy>
T lgamma_imp(T z, const Policy& pol, const lanczos::undefined_lanczos&, int* sign = 0);

template <class T, class Policy>
T gamma_imp(T z, const Policy& pol, const lanczos::undefined_lanczos&)
{
BOOST_MATH_STD_USING

static const char* function = "boost::math::tgamma<%1%>(%1%)";

const bool is_at_zero = (z == 0);

if((boost::math::isnan)(z) || (is_at_zero) || ((boost::math::isinf)(z) && (z < 0)))
return policies::raise_domain_error<T>(function, "Evaluation of tgamma at %1%.", z, pol);

const bool b_neg = (z < 0);

const bool floor_of_z_is_equal_to_z = (floor(z) == z);

if((!b_neg) && floor_of_z_is_equal_to_z && (z < boost::math::max_factorial<T>::value))
{
return boost::math::unchecked_factorial<T>(itrunc(z) - 1);
}

T zz((!b_neg) ? z : -z);

if(zz < tools::cbrt_epsilon<T>())
{
const T a0(1);
const T a1(boost::math::constants::euler<T>());
const T six_euler_squared((boost::math::constants::euler<T>() * boost::math::constants::euler<T>()) * 6);
const T a2((six_euler_squared -  boost::math::constants::pi_sqr<T>()) / 12);

const T inverse_tgamma_series = z * ((a2 * z + a1) * z + a0);

return 1 / inverse_tgamma_series;
}

const int min_arg_for_recursion = minimum_argument_for_bernoulli_recursion<T>();

int n_recur;

if(zz < min_arg_for_recursion)
{
n_recur = boost::math::itrunc(min_arg_for_recursion - zz) + 1;

zz += n_recur;
}
else
{
n_recur = 0;
}
if (!n_recur)
{
if (zz > tools::log_max_value<T>())
return policies::raise_overflow_error<T>(function, 0, pol);
if (log(zz) * zz / 2 > tools::log_max_value<T>())
return policies::raise_overflow_error<T>(function, 0, pol);
}
T gamma_value = scaled_tgamma_no_lanczos(zz, pol);
T power_term = pow(zz, zz / 2);
T exp_term = exp(-zz);
gamma_value *= (power_term * exp_term);
if(!n_recur && (tools::max_value<T>() / power_term < gamma_value))
return policies::raise_overflow_error<T>(function, 0, pol);
gamma_value *= power_term;

if(n_recur)
{
zz = fabs(z) + 1;
for(int k = 1; k < n_recur; ++k)
{
gamma_value /= zz;
zz += 1;
}
gamma_value /= fabs(z);
}

if(b_neg)
{

if(floor_of_z_is_equal_to_z)
return policies::raise_pole_error<T>(function, "Evaluation of tgamma at a negative integer %1%.", z, pol);

gamma_value *= sinpx(z);

BOOST_MATH_INSTRUMENT_VARIABLE(gamma_value);

const bool result_is_too_large_to_represent = (   (abs(gamma_value) < 1)
&& ((tools::max_value<T>() * abs(gamma_value)) < boost::math::constants::pi<T>()));

if(result_is_too_large_to_represent)
return policies::raise_overflow_error<T>(function, "Result of tgamma is too large to represent.", pol);

gamma_value = -boost::math::constants::pi<T>() / gamma_value;
BOOST_MATH_INSTRUMENT_VARIABLE(gamma_value);

if(gamma_value == 0)
return policies::raise_underflow_error<T>(function, "Result of tgamma is too small to represent.", pol);

if((boost::math::fpclassify)(gamma_value) == static_cast<int>(FP_SUBNORMAL))
return policies::raise_denorm_error<T>(function, "Result of tgamma is denormalized.", gamma_value, pol);
}

return gamma_value;
}

template <class T, class Policy>
inline T log_gamma_near_1(const T& z, Policy const& pol)
{
BOOST_MATH_STD_USING 

BOOST_ASSERT(fabs(z) < 1);

T result = -constants::euler<T>() * z;

T power_term = z * z / 2;
int n = 2;
T term = 0;

do
{
term = power_term * boost::math::polygamma(n - 1, T(1), pol);
result += term;
++n;
power_term *= z / n;
} while (fabs(result) * tools::epsilon<T>() < fabs(term));

return result;
}

template <class T, class Policy>
T lgamma_imp(T z, const Policy& pol, const lanczos::undefined_lanczos&, int* sign)
{
BOOST_MATH_STD_USING

static const char* function = "boost::math::lgamma<%1%>(%1%)";

const bool is_at_zero = (z == 0);

if(is_at_zero)
return policies::raise_domain_error<T>(function, "Evaluation of lgamma at zero %1%.", z, pol);
if((boost::math::isnan)(z))
return policies::raise_domain_error<T>(function, "Evaluation of lgamma at %1%.", z, pol);
if((boost::math::isinf)(z))
return policies::raise_overflow_error<T>(function, 0, pol);

const bool b_neg = (z < 0);

const bool floor_of_z_is_equal_to_z = (floor(z) == z);

if((!b_neg) && floor_of_z_is_equal_to_z && (z < boost::math::max_factorial<T>::value))
{
if (sign)
*sign = 1;
return log(boost::math::unchecked_factorial<T>(itrunc(z) - 1));
}

T zz((!b_neg) ? z : -z);

const int min_arg_for_recursion = minimum_argument_for_bernoulli_recursion<T>();

T log_gamma_value;

if (zz < min_arg_for_recursion)
{
if (sign)
* sign = 1;
if(fabs(z - 1) < 0.25)
{
log_gamma_value = log_gamma_near_1(T(zz - 1), pol);
}
else if(fabs(z - 2) < 0.25)
{
log_gamma_value = log_gamma_near_1(T(zz - 2), pol) + log(zz - 1);
}
else if (z > -tools::root_epsilon<T>())
{
if (sign)
*sign = z < 0 ? -1 : 1;
return log(abs(gamma_imp(z, pol, lanczos::undefined_lanczos())));
}
else
{
T g = gamma_imp(zz, pol, lanczos::undefined_lanczos());
if (sign)
{
*sign = g < 0 ? -1 : 1;
}
log_gamma_value = log(abs(g));
}
}
else
{
T sum = scaled_tgamma_no_lanczos(zz, pol, true);
log_gamma_value = zz * (log(zz) - 1) + sum;
}

int sign_of_result = 1;

if(b_neg)
{

if(floor_of_z_is_equal_to_z)
return policies::raise_pole_error<T>(function, "Evaluation of lgamma at a negative integer %1%.", z, pol);

T t = sinpx(z);

if(t < 0)
{
t = -t;
}
else
{
sign_of_result = -sign_of_result;
}

log_gamma_value = - log_gamma_value
+ log(boost::math::constants::pi<T>())
- log(t);
}

if(sign != static_cast<int*>(0U)) { *sign = sign_of_result; }

return log_gamma_value;
}

template <class T, class Policy, class Lanczos>
T tgammap1m1_imp(T dz, Policy const& pol, const Lanczos& l)
{
BOOST_MATH_STD_USING

typedef typename policies::precision<T,Policy>::type precision_type;

typedef boost::integral_constant<int,
precision_type::value <= 0 ? 0 :
precision_type::value <= 64 ? 64 :
precision_type::value <= 113 ? 113 : 0
> tag_type;

T result;
if(dz < 0)
{
if(dz < -0.5)
{
result = boost::math::tgamma(1+dz, pol) - 1;
BOOST_MATH_INSTRUMENT_CODE(result);
}
else
{
result = boost::math::expm1(-boost::math::log1p(dz, pol) 
+ lgamma_small_imp<T>(dz+2, dz + 1, dz, tag_type(), pol, l), pol);
BOOST_MATH_INSTRUMENT_CODE(result);
}
}
else
{
if(dz < 2)
{
result = boost::math::expm1(lgamma_small_imp<T>(dz+1, dz, dz-1, tag_type(), pol, l), pol);
BOOST_MATH_INSTRUMENT_CODE(result);
}
else
{
result = boost::math::tgamma(1+dz, pol) - 1;
BOOST_MATH_INSTRUMENT_CODE(result);
}
}

return result;
}

template <class T, class Policy>
inline T tgammap1m1_imp(T z, Policy const& pol,
const ::boost::math::lanczos::undefined_lanczos&)
{
BOOST_MATH_STD_USING 

if(fabs(z) < 0.55)
{
return boost::math::expm1(log_gamma_near_1(z, pol));
}
return boost::math::expm1(boost::math::lgamma(1 + z, pol));
}

template <class T>
struct small_gamma2_series
{
typedef T result_type;

small_gamma2_series(T a_, T x_) : result(-x_), x(-x_), apn(a_+1), n(1){}

T operator()()
{
T r = result / (apn);
result *= x;
result /= ++n;
apn += 1;
return r;
}

private:
T result, x, apn;
int n;
};
template <class T, class Policy>
T full_igamma_prefix(T a, T z, const Policy& pol)
{
BOOST_MATH_STD_USING

T prefix;
if (z > tools::max_value<T>())
return 0;
T alz = a * log(z);

if(z >= 1)
{
if((alz < tools::log_max_value<T>()) && (-z > tools::log_min_value<T>()))
{
prefix = pow(z, a) * exp(-z);
}
else if(a >= 1)
{
prefix = pow(z / exp(z/a), a);
}
else
{
prefix = exp(alz - z);
}
}
else
{
if(alz > tools::log_min_value<T>())
{
prefix = pow(z, a) * exp(-z);
}
else if(z/a < tools::log_max_value<T>())
{
prefix = pow(z / exp(z/a), a);
}
else
{
prefix = exp(alz - z);
}
}
if((boost::math::fpclassify)(prefix) == (int)FP_INFINITE)
return policies::raise_overflow_error<T>("boost::math::detail::full_igamma_prefix<%1%>(%1%, %1%)", "Result of incomplete gamma function is too large to represent.", pol);

return prefix;
}
template <class T, class Policy, class Lanczos>
T regularised_gamma_prefix(T a, T z, const Policy& pol, const Lanczos& l)
{
BOOST_MATH_STD_USING
if (z >= tools::max_value<T>())
return 0;
T agh = a + static_cast<T>(Lanczos::g()) - T(0.5);
T prefix;
T d = ((z - a) - static_cast<T>(Lanczos::g()) + T(0.5)) / agh;

if(a < 1)
{
if(z <= tools::log_min_value<T>())
{
return exp(a * log(z) - z - lgamma_imp(a, pol, l));
}
else
{
return pow(z, a) * exp(-z) / gamma_imp(a, pol, l);
}
}
else if((fabs(d*d*a) <= 100) && (a > 150))
{
prefix = a * boost::math::log1pmx(d, pol) + z * static_cast<T>(0.5 - Lanczos::g()) / agh;
prefix = exp(prefix);
}
else
{
T alz = a * log(z / agh);
T amz = a - z;
if(((std::min)(alz, amz) <= tools::log_min_value<T>()) || ((std::max)(alz, amz) >= tools::log_max_value<T>()))
{
T amza = amz / a;
if(((std::min)(alz, amz)/2 > tools::log_min_value<T>()) && ((std::max)(alz, amz)/2 < tools::log_max_value<T>()))
{
T sq = pow(z / agh, a / 2) * exp(amz / 2);
prefix = sq * sq;
}
else if(((std::min)(alz, amz)/4 > tools::log_min_value<T>()) && ((std::max)(alz, amz)/4 < tools::log_max_value<T>()) && (z > a))
{
T sq = pow(z / agh, a / 4) * exp(amz / 4);
prefix = sq * sq;
prefix *= prefix;
}
else if((amza > tools::log_min_value<T>()) && (amza < tools::log_max_value<T>()))
{
prefix = pow((z * exp(amza)) / agh, a);
}
else
{
prefix = exp(alz + amz);
}
}
else
{
prefix = pow(z / agh, a) * exp(amz);
}
}
prefix *= sqrt(agh / boost::math::constants::e<T>()) / Lanczos::lanczos_sum_expG_scaled(a);
return prefix;
}
template <class T, class Policy>
T regularised_gamma_prefix(T a, T z, const Policy& pol, const lanczos::undefined_lanczos& l)
{
BOOST_MATH_STD_USING

if((a < 1) && (z < 1))
{
return pow(z, a) * exp(-z) / boost::math::tgamma(a, pol);
}
else if(a > minimum_argument_for_bernoulli_recursion<T>())
{
T scaled_gamma = scaled_tgamma_no_lanczos(a, pol);
T power_term = pow(z / a, a / 2);
T a_minus_z = a - z;
if ((0 == power_term) || (fabs(a_minus_z) > tools::log_max_value<T>()))
{
return exp(a * log(z / a) + a_minus_z - log(scaled_gamma));
}
return (power_term * exp(a_minus_z)) * (power_term / scaled_gamma);
}
else
{
const int min_z = minimum_argument_for_bernoulli_recursion<T>();
long shift = 1 + ltrunc(min_z - a);
T result = regularised_gamma_prefix(T(a + shift), z, pol, l);
if (result != 0)
{
for (long i = 0; i < shift; ++i)
{
result /= z;
result *= a + i;
}
return result;
}
else
{
T scaled_gamma = scaled_tgamma_no_lanczos(T(a + shift), pol);
T power_term_1 = pow(z / (a + shift), a);
T power_term_2 = pow(a + shift, -shift);
T power_term_3 = exp(a + shift - z);
if ((0 == power_term_1) || (0 == power_term_2) || (0 == power_term_3) || (fabs(a + shift - z) > tools::log_max_value<T>()))
{
return exp(a * log(z) - z - boost::math::lgamma(a, pol));
}
result = power_term_1 * power_term_2 * power_term_3 / scaled_gamma;
for (long i = 0; i < shift; ++i)
{
result *= a + i;
}
return result;
}
}
}
template <class T, class Policy>
inline T tgamma_small_upper_part(T a, T x, const Policy& pol, T* pgam = 0, bool invert = false, T* pderivative = 0)
{
BOOST_MATH_STD_USING  
T result;
result = boost::math::tgamma1pm1(a, pol);
if(pgam)
*pgam = (result + 1) / a;
T p = boost::math::powm1(x, a, pol);
result -= p;
result /= a;
detail::small_gamma2_series<T> s(a, x);
boost::uintmax_t max_iter = policies::get_max_series_iterations<Policy>() - 10;
p += 1;
if(pderivative)
*pderivative = p / (*pgam * exp(x));
T init_value = invert ? *pgam : 0;
result = -p * tools::sum_series(s, boost::math::policies::get_epsilon<T, Policy>(), max_iter, (init_value - result) / p);
policies::check_series_iterations<T>("boost::math::tgamma_small_upper_part<%1%>(%1%, %1%)", max_iter, pol);
if(invert)
result = -result;
return result;
}
template <class T, class Policy>
inline T finite_gamma_q(T a, T x, Policy const& pol, T* pderivative = 0)
{
BOOST_MATH_STD_USING
T e = exp(-x);
T sum = e;
if(sum != 0)
{
T term = sum;
for(unsigned n = 1; n < a; ++n)
{
term /= n;
term *= x;
sum += term;
}
}
if(pderivative)
{
*pderivative = e * pow(x, a) / boost::math::unchecked_factorial<T>(itrunc(T(a - 1), pol));
}
return sum;
}
template <class T, class Policy>
T finite_half_gamma_q(T a, T x, T* p_derivative, const Policy& pol)
{
BOOST_MATH_STD_USING
T e = boost::math::erfc(sqrt(x), pol);
if((e != 0) && (a > 1))
{
T term = exp(-x) / sqrt(constants::pi<T>() * x);
term *= x;
static const T half = T(1) / 2;
term /= half;
T sum = term;
for(unsigned n = 2; n < a; ++n)
{
term /= n - half;
term *= x;
sum += term;
}
e += sum;
if(p_derivative)
{
*p_derivative = 0;
}
}
else if(p_derivative)
{
*p_derivative = sqrt(x) * exp(-x) / constants::root_pi<T>();
}
return e;
}
template <class T>
struct incomplete_tgamma_large_x_series
{
typedef T result_type;
incomplete_tgamma_large_x_series(const T& a, const T& x)
: a_poch(a - 1), z(x), term(1) {}
T operator()()
{
T result = term;
term *= a_poch / z;
a_poch -= 1;
return result;
}
T a_poch, z, term;
};

template <class T, class Policy>
T incomplete_tgamma_large_x(const T& a, const T& x, const Policy& pol)
{
BOOST_MATH_STD_USING
incomplete_tgamma_large_x_series<T> s(a, x);
boost::uintmax_t max_iter = boost::math::policies::get_max_series_iterations<Policy>();
T result = boost::math::tools::sum_series(s, boost::math::policies::get_epsilon<T, Policy>(), max_iter);
boost::math::policies::check_series_iterations<T>("boost::math::tgamma<%1%>(%1%,%1%)", max_iter, pol);
return result;
}


template <class T, class Policy>
T gamma_incomplete_imp(T a, T x, bool normalised, bool invert, 
const Policy& pol, T* p_derivative)
{
static const char* function = "boost::math::gamma_p<%1%>(%1%, %1%)";
if(a <= 0)
return policies::raise_domain_error<T>(function, "Argument a to the incomplete gamma function must be greater than zero (got a=%1%).", a, pol);
if(x < 0)
return policies::raise_domain_error<T>(function, "Argument x to the incomplete gamma function must be >= 0 (got x=%1%).", x, pol);

BOOST_MATH_STD_USING

typedef typename lanczos::lanczos<T, Policy>::type lanczos_type;

T result = 0; 

if(a >= max_factorial<T>::value && !normalised)
{
if(invert && (a * 4 < x))
{
result = a * log(x) - x;
if(p_derivative)
*p_derivative = exp(result);
result += log(upper_gamma_fraction(a, x, policies::get_epsilon<T, Policy>()));
}
else if(!invert && (a > 4 * x))
{
result = a * log(x) - x;
if(p_derivative)
*p_derivative = exp(result);
T init_value = 0;
result += log(detail::lower_gamma_series(a, x, pol, init_value) / a);
}
else
{
result = gamma_incomplete_imp(a, x, true, invert, pol, p_derivative);
if(result == 0)
{
if(invert)
{
result = 1 + 1 / (12 * a) + 1 / (288 * a * a);
result = log(result) - a + (a - 0.5f) * log(a) + log(boost::math::constants::root_two_pi<T>());
if(p_derivative)
*p_derivative = exp(a * log(x) - x);
}
else
{
result = a * log(x) - x;
if(p_derivative)
*p_derivative = exp(result);
T init_value = 0;
result += log(detail::lower_gamma_series(a, x, pol, init_value) / a);
}
}
else
{
result = log(result) + boost::math::lgamma(a, pol);
}
}
if(result > tools::log_max_value<T>())
return policies::raise_overflow_error<T>(function, 0, pol);
return exp(result);
}

BOOST_ASSERT((p_derivative == 0) || normalised);

bool is_int, is_half_int;
bool is_small_a = (a < 30) && (a <= x + 1) && (x < tools::log_max_value<T>());
if(is_small_a)
{
T fa = floor(a);
is_int = (fa == a);
is_half_int = is_int ? false : (fabs(fa - a) == 0.5f);
}
else
{
is_int = is_half_int = false;
}

int eval_method;

if(is_int && (x > 0.6))
{
invert = !invert;
eval_method = 0;
}
else if(is_half_int && (x > 0.2))
{
invert = !invert;
eval_method = 1;
}
else if((x < tools::root_epsilon<T>()) && (a > 1))
{
eval_method = 6;
}
else if ((x > 1000) && ((a < x) || (fabs(a - 50) / x < 1)))
{
invert = !invert;
eval_method = 7;
}
else if(x < 0.5)
{
if(-0.4 / log(x) < a)
{
eval_method = 2;
}
else
{
eval_method = 3;
}
}
else if(x < 1.1)
{
if(x * 0.75f < a)
{
eval_method = 2;
}
else
{
eval_method = 3;
}
}
else
{
bool use_temme = false;
if(normalised && std::numeric_limits<T>::is_specialized && (a > 20))
{
T sigma = fabs((x-a)/a);
if((a > 200) && (policies::digits<T, Policy>() <= 113))
{
if(20 / a > sigma * sigma)
use_temme = true;
}
else if(policies::digits<T, Policy>() <= 64)
{
if(sigma < 0.4)
use_temme = true;
}
}
if(use_temme)
{
eval_method = 5;
}
else
{
if(x - (1 / (3 * x)) < a)
{
eval_method = 2;
}
else
{
eval_method = 4;
invert = !invert;
}
}
}

switch(eval_method)
{
case 0:
{
result = finite_gamma_q(a, x, pol, p_derivative);
if(!normalised)
result *= boost::math::tgamma(a, pol);
break;
}
case 1:
{
result = finite_half_gamma_q(a, x, p_derivative, pol);
if(!normalised)
result *= boost::math::tgamma(a, pol);
if(p_derivative && (*p_derivative == 0))
*p_derivative = regularised_gamma_prefix(a, x, pol, lanczos_type());
break;
}
case 2:
{
result = normalised ? regularised_gamma_prefix(a, x, pol, lanczos_type()) : full_igamma_prefix(a, x, pol);
if(p_derivative)
*p_derivative = result;
if(result != 0)
{
T init_value = 0;
bool optimised_invert = false;
if(invert)
{
init_value = (normalised ? 1 : boost::math::tgamma(a, pol));
if(normalised || (result >= 1) || (tools::max_value<T>() * result > init_value))
{
init_value /= result;
if(normalised || (a < 1) || (tools::max_value<T>() / a > init_value))
{
init_value *= -a;
optimised_invert = true;
}
else
init_value = 0;
}
else
init_value = 0;
}
result *= detail::lower_gamma_series(a, x, pol, init_value) / a;
if(optimised_invert)
{
invert = false;
result = -result;
}
}
break;
}
case 3:
{
invert = !invert;
T g;
result = tgamma_small_upper_part(a, x, pol, &g, invert, p_derivative);
invert = false;
if(normalised)
result /= g;
break;
}
case 4:
{
result = normalised ? regularised_gamma_prefix(a, x, pol, lanczos_type()) : full_igamma_prefix(a, x, pol);
if(p_derivative)
*p_derivative = result;
if(result != 0)
result *= upper_gamma_fraction(a, x, policies::get_epsilon<T, Policy>());
break;
}
case 5:
{
typedef typename policies::precision<T, Policy>::type precision_type;

typedef boost::integral_constant<int,
precision_type::value <= 0 ? 0 :
precision_type::value <= 53 ? 53 :
precision_type::value <= 64 ? 64 :
precision_type::value <= 113 ? 113 : 0
> tag_type;

result = igamma_temme_large(a, x, pol, static_cast<tag_type const*>(0));
if(x >= a)
invert = !invert;
if(p_derivative)
*p_derivative = regularised_gamma_prefix(a, x, pol, lanczos_type());
break;
}
case 6:
{
if(!normalised)
result = pow(x, a) / (a);
else
{
#ifndef BOOST_NO_EXCEPTIONS
try 
{
result = pow(x, a) / boost::math::tgamma(a + 1, pol);
}
catch (const std::overflow_error&)
{
result = 0;
}
#else
result = pow(x, a) / boost::math::tgamma(a + 1, pol);
#endif
}
result *= 1 - a * x / (a + 1);
if (p_derivative)
*p_derivative = regularised_gamma_prefix(a, x, pol, lanczos_type());
break;
}
case 7:
{
result = normalised ? regularised_gamma_prefix(a, x, pol, lanczos_type()) : full_igamma_prefix(a, x, pol);
if (p_derivative)
*p_derivative = result;
result /= x;
if (result != 0)
result *= incomplete_tgamma_large_x(a, x, pol);
break;
}
}

if(normalised && (result > 1))
result = 1;
if(invert)
{
T gam = normalised ? 1 : boost::math::tgamma(a, pol);
result = gam - result;
}
if(p_derivative)
{
if((x < 1) && (tools::max_value<T>() * x < *p_derivative))
{
*p_derivative = tools::max_value<T>() / 2;
}

*p_derivative /= x;
}

return result;
}

template <class T, class Policy, class Lanczos>
T tgamma_delta_ratio_imp_lanczos(T z, T delta, const Policy& pol, const Lanczos& l)
{
BOOST_MATH_STD_USING
if(z < tools::epsilon<T>())
{
if(boost::math::max_factorial<T>::value < delta)
{
T ratio = tgamma_delta_ratio_imp_lanczos(delta, T(boost::math::max_factorial<T>::value - delta), pol, l);
ratio *= z;
ratio *= boost::math::unchecked_factorial<T>(boost::math::max_factorial<T>::value - 1);
return 1 / ratio;
}
else
{
return 1 / (z * boost::math::tgamma(z + delta, pol));
}
}
T zgh = static_cast<T>(z + Lanczos::g() - constants::half<T>());
T result;
if(z + delta == z)
{
if(fabs(delta) < 10)
result = exp((constants::half<T>() - z) * boost::math::log1p(delta / zgh, pol));
else
result = 1;
}
else
{
if(fabs(delta) < 10)
{
result = exp((constants::half<T>() - z) * boost::math::log1p(delta / zgh, pol));
}
else
{
result = pow(zgh / (zgh + delta), z - constants::half<T>());
}
result *= Lanczos::lanczos_sum(z) / Lanczos::lanczos_sum(T(z + delta));
}
result *= pow(constants::e<T>() / (zgh + delta), delta);
return result;
}
template <class T, class Policy>
T tgamma_delta_ratio_imp_lanczos(T z, T delta, const Policy& pol, const lanczos::undefined_lanczos& l)
{
BOOST_MATH_STD_USING

long numerator_shift = 0;
long denominator_shift = 0;
const int min_z = minimum_argument_for_bernoulli_recursion<T>();

if (min_z > z)
numerator_shift = 1 + ltrunc(min_z - z);
if (min_z > z + delta)
denominator_shift = 1 + ltrunc(min_z - z - delta);
if (numerator_shift == 0 && denominator_shift == 0)
{
T scaled_tgamma_num = scaled_tgamma_no_lanczos(z, pol);
T scaled_tgamma_denom = scaled_tgamma_no_lanczos(T(z + delta), pol);
T result = scaled_tgamma_num / scaled_tgamma_denom;
result *= exp(z * boost::math::log1p(-delta / (z + delta), pol)) * pow((delta + z) / constants::e<T>(), -delta);
return result;
}
T zz = z + numerator_shift;
T dd = delta - (numerator_shift - denominator_shift);
T ratio = tgamma_delta_ratio_imp_lanczos(zz, dd, pol, l);
for (long long i = 0; i < numerator_shift; ++i)
{
ratio /= (z + i);
if (i < denominator_shift)
ratio *= (z + delta + i);
}
for (long long i = numerator_shift; i < denominator_shift; ++i)
{
ratio *= (z + delta + i);
}
return ratio;
}

template <class T, class Policy>
T tgamma_delta_ratio_imp(T z, T delta, const Policy& pol)
{
BOOST_MATH_STD_USING

if((z <= 0) || (z + delta <= 0))
{
return boost::math::tgamma(z, pol) / boost::math::tgamma(z + delta, pol);
}

if(floor(delta) == delta)
{
if(floor(z) == z)
{
if((z <= max_factorial<T>::value) && (z + delta <= max_factorial<T>::value))
{
return unchecked_factorial<T>((unsigned)itrunc(z, pol) - 1) / unchecked_factorial<T>((unsigned)itrunc(T(z + delta), pol) - 1);
}
}
if(fabs(delta) < 20)
{
if(delta == 0)
return 1;
if(delta < 0)
{
z -= 1;
T result = z;
while(0 != (delta += 1))
{
z -= 1;
result *= z;
}
return result;
}
else
{
T result = 1 / z;
while(0 != (delta -= 1))
{
z += 1;
result /= z;
}
return result;
}
}
}
typedef typename lanczos::lanczos<T, Policy>::type lanczos_type;
return tgamma_delta_ratio_imp_lanczos(z, delta, pol, lanczos_type());
}

template <class T, class Policy>
T tgamma_ratio_imp(T x, T y, const Policy& pol)
{
BOOST_MATH_STD_USING

if((x <= 0) || (boost::math::isinf)(x))
return policies::raise_domain_error<T>("boost::math::tgamma_ratio<%1%>(%1%, %1%)", "Gamma function ratios only implemented for positive arguments (got a=%1%).", x, pol);
if((y <= 0) || (boost::math::isinf)(y))
return policies::raise_domain_error<T>("boost::math::tgamma_ratio<%1%>(%1%, %1%)", "Gamma function ratios only implemented for positive arguments (got b=%1%).", y, pol);

if(x <= tools::min_value<T>())
{
T shift = ldexp(T(1), tools::digits<T>());
return shift * tgamma_ratio_imp(T(x * shift), y, pol);
}

if((x < max_factorial<T>::value) && (y < max_factorial<T>::value))
{
return boost::math::tgamma(x, pol) / boost::math::tgamma(y, pol);
}
T prefix = 1;
if(x < 1)
{
if(y < 2 * max_factorial<T>::value)
{
prefix /= x;
x += 1;
while(y >=  max_factorial<T>::value)
{
y -= 1;
prefix /= y;
}
return prefix * boost::math::tgamma(x, pol) / boost::math::tgamma(y, pol);
}
return exp(boost::math::lgamma(x, pol) - boost::math::lgamma(y, pol));
}
if(y < 1)
{
if(x < 2 * max_factorial<T>::value)
{
prefix *= y;
y += 1;
while(x >= max_factorial<T>::value)
{
x -= 1;
prefix *= x;
}
return prefix * boost::math::tgamma(x, pol) / boost::math::tgamma(y, pol);
}
return exp(boost::math::lgamma(x, pol) - boost::math::lgamma(y, pol));
}
return boost::math::tgamma_delta_ratio(x, y - x, pol);
}

template <class T, class Policy>
T gamma_p_derivative_imp(T a, T x, const Policy& pol)
{
BOOST_MATH_STD_USING
if(a <= 0)
return policies::raise_domain_error<T>("boost::math::gamma_p_derivative<%1%>(%1%, %1%)", "Argument a to the incomplete gamma function must be greater than zero (got a=%1%).", a, pol);
if(x < 0)
return policies::raise_domain_error<T>("boost::math::gamma_p_derivative<%1%>(%1%, %1%)", "Argument x to the incomplete gamma function must be >= 0 (got x=%1%).", x, pol);
if(x == 0)
{
return (a > 1) ? 0 :
(a == 1) ? 1 : policies::raise_overflow_error<T>("boost::math::gamma_p_derivative<%1%>(%1%, %1%)", 0, pol);
}
typedef typename lanczos::lanczos<T, Policy>::type lanczos_type;
T f1 = detail::regularised_gamma_prefix(a, x, pol, lanczos_type());
if((x < 1) && (tools::max_value<T>() * x < f1))
{
return policies::raise_overflow_error<T>("boost::math::gamma_p_derivative<%1%>(%1%, %1%)", 0, pol);
}
if(f1 == 0)
{
f1 = a * log(x) - x - lgamma(a, pol) - log(x);
f1 = exp(f1);
}
else
f1 /= x;

return f1;
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type 
tgamma(T z, const Policy& , const boost::true_type)
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename lanczos::lanczos<value_type, Policy>::type evaluation_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;
return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::gamma_imp(static_cast<value_type>(z), forwarding_policy(), evaluation_type()), "boost::math::tgamma<%1%>(%1%)");
}

template <class T, class Policy>
struct igamma_initializer
{
struct init
{
init()
{
typedef typename policies::precision<T, Policy>::type precision_type;

typedef boost::integral_constant<int,
precision_type::value <= 0 ? 0 :
precision_type::value <= 53 ? 53 :
precision_type::value <= 64 ? 64 :
precision_type::value <= 113 ? 113 : 0
> tag_type;

do_init(tag_type());
}
template <int N>
static void do_init(const boost::integral_constant<int, N>&)
{
if(std::numeric_limits<T>::digits)
{
boost::math::gamma_p(static_cast<T>(400), static_cast<T>(400), Policy());
}
}
static void do_init(const boost::integral_constant<int, 53>&){}
void force_instantiate()const{}
};
static const init initializer;
static void force_instantiate()
{
initializer.force_instantiate();
}
};

template <class T, class Policy>
const typename igamma_initializer<T, Policy>::init igamma_initializer<T, Policy>::initializer;

template <class T, class Policy>
struct lgamma_initializer
{
struct init
{
init()
{
typedef typename policies::precision<T, Policy>::type precision_type;
typedef boost::integral_constant<int,
precision_type::value <= 0 ? 0 :
precision_type::value <= 64 ? 64 :
precision_type::value <= 113 ? 113 : 0
> tag_type;

do_init(tag_type());
}
static void do_init(const boost::integral_constant<int, 64>&)
{
boost::math::lgamma(static_cast<T>(2.5), Policy());
boost::math::lgamma(static_cast<T>(1.25), Policy());
boost::math::lgamma(static_cast<T>(1.75), Policy());
}
static void do_init(const boost::integral_constant<int, 113>&)
{
boost::math::lgamma(static_cast<T>(2.5), Policy());
boost::math::lgamma(static_cast<T>(1.25), Policy());
boost::math::lgamma(static_cast<T>(1.5), Policy());
boost::math::lgamma(static_cast<T>(1.75), Policy());
}
static void do_init(const boost::integral_constant<int, 0>&)
{
}
void force_instantiate()const{}
};
static const init initializer;
static void force_instantiate()
{
initializer.force_instantiate();
}
};

template <class T, class Policy>
const typename lgamma_initializer<T, Policy>::init lgamma_initializer<T, Policy>::initializer;

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type
tgamma(T1 a, T2 z, const Policy&, const boost::false_type)
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

igamma_initializer<value_type, forwarding_policy>::force_instantiate();

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::gamma_incomplete_imp(static_cast<value_type>(a),
static_cast<value_type>(z), false, true,
forwarding_policy(), static_cast<value_type*>(0)), "boost::math::tgamma<%1%>(%1%, %1%)");
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type
tgamma(T1 a, T2 z, const boost::false_type& tag)
{
return tgamma(a, z, policies::policy<>(), tag);
}


} 

template <class T>
inline typename tools::promote_args<T>::type 
tgamma(T z)
{
return tgamma(z, policies::policy<>());
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type 
lgamma(T z, int* sign, const Policy&)
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename lanczos::lanczos<value_type, Policy>::type evaluation_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

detail::lgamma_initializer<value_type, forwarding_policy>::force_instantiate();

return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::lgamma_imp(static_cast<value_type>(z), forwarding_policy(), evaluation_type(), sign), "boost::math::lgamma<%1%>(%1%)");
}

template <class T>
inline typename tools::promote_args<T>::type 
lgamma(T z, int* sign)
{
return lgamma(z, sign, policies::policy<>());
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type 
lgamma(T x, const Policy& pol)
{
return ::boost::math::lgamma(x, 0, pol);
}

template <class T>
inline typename tools::promote_args<T>::type 
lgamma(T x)
{
return ::boost::math::lgamma(x, 0, policies::policy<>());
}

template <class T, class Policy>
inline typename tools::promote_args<T>::type 
tgamma1pm1(T z, const Policy& )
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename lanczos::lanczos<value_type, Policy>::type evaluation_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

return policies::checked_narrowing_cast<typename remove_cv<result_type>::type, forwarding_policy>(detail::tgammap1m1_imp(static_cast<value_type>(z), forwarding_policy(), evaluation_type()), "boost::math::tgamma1pm1<%!%>(%1%)");
}

template <class T>
inline typename tools::promote_args<T>::type 
tgamma1pm1(T z)
{
return tgamma1pm1(z, policies::policy<>());
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type
tgamma(T1 a, T2 z)
{
typedef typename policies::is_policy<T2>::type maybe_policy;
return detail::tgamma(a, z, maybe_policy());
}
template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type
tgamma(T1 a, T2 z, const Policy& pol)
{
return detail::tgamma(a, z, pol, boost::false_type());
}
template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type
tgamma_lower(T1 a, T2 z, const Policy&)
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

detail::igamma_initializer<value_type, forwarding_policy>::force_instantiate();

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::gamma_incomplete_imp(static_cast<value_type>(a),
static_cast<value_type>(z), false, false,
forwarding_policy(), static_cast<value_type*>(0)), "tgamma_lower<%1%>(%1%, %1%)");
}
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type
tgamma_lower(T1 a, T2 z)
{
return tgamma_lower(a, z, policies::policy<>());
}
template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type
gamma_q(T1 a, T2 z, const Policy& )
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

detail::igamma_initializer<value_type, forwarding_policy>::force_instantiate();

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::gamma_incomplete_imp(static_cast<value_type>(a),
static_cast<value_type>(z), true, true,
forwarding_policy(), static_cast<value_type*>(0)), "gamma_q<%1%>(%1%, %1%)");
}
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type
gamma_q(T1 a, T2 z)
{
return gamma_q(a, z, policies::policy<>());
}
template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type
gamma_p(T1 a, T2 z, const Policy&)
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

detail::igamma_initializer<value_type, forwarding_policy>::force_instantiate();

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::gamma_incomplete_imp(static_cast<value_type>(a),
static_cast<value_type>(z), true, false,
forwarding_policy(), static_cast<value_type*>(0)), "gamma_p<%1%>(%1%, %1%)");
}
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type
gamma_p(T1 a, T2 z)
{
return gamma_p(a, z, policies::policy<>());
}

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type 
tgamma_delta_ratio(T1 z, T2 delta, const Policy& )
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::tgamma_delta_ratio_imp(static_cast<value_type>(z), static_cast<value_type>(delta), forwarding_policy()), "boost::math::tgamma_delta_ratio<%1%>(%1%, %1%)");
}
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type 
tgamma_delta_ratio(T1 z, T2 delta)
{
return tgamma_delta_ratio(z, delta, policies::policy<>());
}
template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type 
tgamma_ratio(T1 a, T2 b, const Policy&)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::tgamma_ratio_imp(static_cast<value_type>(a), static_cast<value_type>(b), forwarding_policy()), "boost::math::tgamma_delta_ratio<%1%>(%1%, %1%)");
}
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type 
tgamma_ratio(T1 a, T2 b)
{
return tgamma_ratio(a, b, policies::policy<>());
}

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type 
gamma_p_derivative(T1 a, T2 x, const Policy&)
{
BOOST_FPU_EXCEPTION_GUARD
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

return policies::checked_narrowing_cast<result_type, forwarding_policy>(detail::gamma_p_derivative_imp(static_cast<value_type>(a), static_cast<value_type>(x), forwarding_policy()), "boost::math::gamma_p_derivative<%1%>(%1%, %1%)");
}
template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type 
gamma_p_derivative(T1 a, T2 x)
{
return gamma_p_derivative(a, x, policies::policy<>());
}

} 
} 

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#include <boost/math/special_functions/detail/igamma_inverse.hpp>
#include <boost/math/special_functions/detail/gamma_inva.hpp>
#include <boost/math/special_functions/erf.hpp>

#endif 
