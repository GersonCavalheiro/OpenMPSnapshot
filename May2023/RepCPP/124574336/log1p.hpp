
#ifndef BOOST_MATH_LOG1P_INCLUDED
#define BOOST_MATH_LOG1P_INCLUDED

#ifdef _MSC_VER
#pragma once
#pragma warning(push)
#pragma warning(disable:4702) 
#endif

#include <boost/config/no_tr1/cmath.hpp>
#include <math.h> 
#include <boost/limits.hpp>
#include <boost/math/tools/config.hpp>
#include <boost/math/tools/series.hpp>
#include <boost/math/tools/rational.hpp>
#include <boost/math/tools/big_constant.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/math_fwd.hpp>

#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#  include <boost/static_assert.hpp>
#else
#  include <boost/assert.hpp>
#endif

#if defined(__GNUC__) && defined(BOOST_MATH_USE_FLOAT128)
#pragma GCC system_header
#endif

namespace boost{ namespace math{

namespace detail
{
template <class T>
struct log1p_series
{
typedef T result_type;

log1p_series(T x)
: k(0), m_mult(-x), m_prod(-1){}

T operator()()
{
m_prod *= m_mult;
return m_prod / ++k;
}

int count()const
{
return k;
}

private:
int k;
const T m_mult;
T m_prod;
log1p_series(const log1p_series&);
log1p_series& operator=(const log1p_series&);
};

template <class T, class Policy>
T log1p_imp(T const & x, const Policy& pol, const boost::integral_constant<int, 0>&)
{ 
typedef typename tools::promote_args<T>::type result_type;
BOOST_MATH_STD_USING

static const char* function = "boost::math::log1p<%1%>(%1%)";

if((x < -1) || (boost::math::isnan)(x))
return policies::raise_domain_error<T>(
function, "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<T>(
function, 0, pol);

result_type a = abs(result_type(x));
if(a > result_type(0.5f))
return log(1 + result_type(x));
if(a < tools::epsilon<result_type>())
return x;
detail::log1p_series<result_type> s(x);
boost::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !BOOST_WORKAROUND(__EDG_VERSION__, <= 245)
result_type result = tools::sum_series(s, policies::get_epsilon<result_type, Policy>(), max_iter);
#else
result_type zero = 0;
result_type result = tools::sum_series(s, policies::get_epsilon<result_type, Policy>(), max_iter, zero);
#endif
policies::check_series_iterations<T>(function, max_iter, pol);
return result;
}

template <class T, class Policy>
T log1p_imp(T const& x, const Policy& pol, const boost::integral_constant<int, 53>&)
{ 
BOOST_MATH_STD_USING

static const char* function = "boost::math::log1p<%1%>(%1%)";

if(x < -1)
return policies::raise_domain_error<T>(
function, "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<T>(
function, 0, pol);

T a = fabs(x);
if(a > 0.5f)
return log(1 + x);
if(a < tools::epsilon<T>())
return x;

static const T P[] = {    
0.15141069795941984e-16L,
0.35495104378055055e-15L,
0.33333333333332835L,
0.99249063543365859L,
1.1143969784156509L,
0.58052937949269651L,
0.13703234928513215L,
0.011294864812099712L
};
static const T Q[] = {    
1L,
3.7274719063011499L,
5.5387948649720334L,
4.159201143419005L,
1.6423855110312755L,
0.31706251443180914L,
0.022665554431410243L,
-0.29252538135177773e-5L
};

T result = 1 - x / 2 + tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
result *= x;

return result;
}

template <class T, class Policy>
T log1p_imp(T const& x, const Policy& pol, const boost::integral_constant<int, 64>&)
{ 
BOOST_MATH_STD_USING

static const char* function = "boost::math::log1p<%1%>(%1%)";

if(x < -1)
return policies::raise_domain_error<T>(
function, "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<T>(
function, 0, pol);

T a = fabs(x);
if(a > 0.5f)
return log(1 + x);
if(a < tools::epsilon<T>())
return x;

static const T P[] = {    
BOOST_MATH_BIG_CONSTANT(T, 64, -0.807533446680736736712e-19),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.490881544804798926426e-18),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.333333333333333373941),
BOOST_MATH_BIG_CONSTANT(T, 64, 1.17141290782087994162),
BOOST_MATH_BIG_CONSTANT(T, 64, 1.62790522814926264694),
BOOST_MATH_BIG_CONSTANT(T, 64, 1.13156411870766876113),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.408087379932853785336),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.0706537026422828914622),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.00441709903782239229447)
};
static const T Q[] = {    
BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
BOOST_MATH_BIG_CONSTANT(T, 64, 4.26423872346263928361),
BOOST_MATH_BIG_CONSTANT(T, 64, 7.48189472704477708962),
BOOST_MATH_BIG_CONSTANT(T, 64, 6.94757016732904280913),
BOOST_MATH_BIG_CONSTANT(T, 64, 3.6493508622280767304),
BOOST_MATH_BIG_CONSTANT(T, 64, 1.06884863623790638317),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.158292216998514145947),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.00885295524069924328658),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.560026216133415663808e-6)
};

T result = 1 - x / 2 + tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
result *= x;

return result;
}

template <class T, class Policy>
T log1p_imp(T const& x, const Policy& pol, const boost::integral_constant<int, 24>&)
{ 
BOOST_MATH_STD_USING

static const char* function = "boost::math::log1p<%1%>(%1%)";

if(x < -1)
return policies::raise_domain_error<T>(
function, "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<T>(
function, 0, pol);

T a = fabs(x);
if(a > 0.5f)
return log(1 + x);
if(a < tools::epsilon<T>())
return x;

static const T P[] = {    
-0.671192866803148236519e-7L,
0.119670999140731844725e-6L,
0.333339469182083148598L,
0.237827183019664122066L
};
static const T Q[] = {    
1L,
1.46348272586988539733L,
0.497859871350117338894L,
-0.00471666268910169651936L
};

T result = 1 - x / 2 + tools::evaluate_polynomial(P, x) / tools::evaluate_polynomial(Q, x);
result *= x;

return result;
}

template <class T, class Policy, class tag>
struct log1p_initializer
{
struct init
{
init()
{
do_init(tag());
}
template <int N>
static void do_init(const boost::integral_constant<int, N>&){}
static void do_init(const boost::integral_constant<int, 64>&)
{
boost::math::log1p(static_cast<T>(0.25), Policy());
}
void force_instantiate()const{}
};
static const init initializer;
static void force_instantiate()
{
initializer.force_instantiate();
}
};

template <class T, class Policy, class tag>
const typename log1p_initializer<T, Policy, tag>::init log1p_initializer<T, Policy, tag>::initializer;


} 

template <class T, class Policy>
inline typename tools::promote_args<T>::type log1p(T x, const Policy&)
{ 
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::precision<result_type, Policy>::type precision_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

typedef boost::integral_constant<int,
precision_type::value <= 0 ? 0 :
precision_type::value <= 53 ? 53 :
precision_type::value <= 64 ? 64 : 0
> tag_type;

detail::log1p_initializer<value_type, forwarding_policy, tag_type>::force_instantiate();

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::log1p_imp(static_cast<value_type>(x), forwarding_policy(), tag_type()), "boost::math::log1p<%1%>(%1%)");
}

#if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
inline float log1p(float z)
{
return log1p<float>(z);
}
inline double log1p(double z)
{
return log1p<double>(z);
}
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
inline long double log1p(long double z)
{
return log1p<long double>(z);
}
#endif
#endif

#ifdef log1p
#  ifndef BOOST_HAS_LOG1P
#     define BOOST_HAS_LOG1P
#  endif
#  undef log1p
#endif

#if defined(BOOST_HAS_LOG1P) && !(defined(__osf__) && defined(__DECCXX_VER))
#  ifdef BOOST_MATH_USE_C99
template <class Policy>
inline float log1p(float x, const Policy& pol)
{ 
if(x < -1)
return policies::raise_domain_error<float>(
"log1p<%1%>(%1%)", "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<float>(
"log1p<%1%>(%1%)", 0, pol);
return ::log1pf(x); 
}
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
template <class Policy>
inline long double log1p(long double x, const Policy& pol)
{ 
if(x < -1)
return policies::raise_domain_error<long double>(
"log1p<%1%>(%1%)", "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<long double>(
"log1p<%1%>(%1%)", 0, pol);
return ::log1pl(x); 
}
#endif
#else
template <class Policy>
inline float log1p(float x, const Policy& pol)
{ 
if(x < -1)
return policies::raise_domain_error<float>(
"log1p<%1%>(%1%)", "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<float>(
"log1p<%1%>(%1%)", 0, pol);
return ::log1p(x); 
}
#endif
template <class Policy>
inline double log1p(double x, const Policy& pol)
{ 
if(x < -1)
return policies::raise_domain_error<double>(
"log1p<%1%>(%1%)", "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<double>(
"log1p<%1%>(%1%)", 0, pol);
return ::log1p(x); 
}
#elif defined(_MSC_VER) && (BOOST_MSVC >= 1400)
template <class Policy>
inline double log1p(double x, const Policy& pol)
{
if(x < -1)
return policies::raise_domain_error<double>(
"log1p<%1%>(%1%)", "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<double>(
"log1p<%1%>(%1%)", 0, pol);
double u = 1+x;
if(u == 1.0) 
return x; 
else
return ::log(u)*(x/(u-1.0));
}
template <class Policy>
inline float log1p(float x, const Policy& pol)
{
return static_cast<float>(boost::math::log1p(static_cast<double>(x), pol));
}
#ifndef _WIN32_WCE
template <class Policy>
inline long double log1p(long double x, const Policy& pol)
{
if(x < -1)
return policies::raise_domain_error<long double>(
"log1p<%1%>(%1%)", "log1p(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<long double>(
"log1p<%1%>(%1%)", 0, pol);
long double u = 1+x;
if(u == 1.0) 
return x; 
else
return ::logl(u)*(x/(u-1.0));
}
#endif
#endif

template <class T>
inline typename tools::promote_args<T>::type log1p(T x)
{
return boost::math::log1p(x, policies::policy<>());
}
template <class T, class Policy>
inline typename tools::promote_args<T>::type 
log1pmx(T x, const Policy& pol)
{
typedef typename tools::promote_args<T>::type result_type;
BOOST_MATH_STD_USING
static const char* function = "boost::math::log1pmx<%1%>(%1%)";

if(x < -1)
return policies::raise_domain_error<T>(
function, "log1pmx(x) requires x > -1, but got x = %1%.", x, pol);
if(x == -1)
return -policies::raise_overflow_error<T>(
function, 0, pol);

result_type a = abs(result_type(x));
if(a > result_type(0.95f))
return log(1 + result_type(x)) - result_type(x);
if(a < tools::epsilon<result_type>())
return -x * x / 2;
boost::math::detail::log1p_series<T> s(x);
s();
boost::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();
#if BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582))
T zero = 0;
T result = boost::math::tools::sum_series(s, policies::get_epsilon<T, Policy>(), max_iter, zero);
#else
T result = boost::math::tools::sum_series(s, policies::get_epsilon<T, Policy>(), max_iter);
#endif
policies::check_series_iterations<T>(function, max_iter, pol);
return result;
}

template <class T>
inline typename tools::promote_args<T>::type log1pmx(T x)
{
return log1pmx(x, policies::policy<>());
}

} 
} 

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif 



