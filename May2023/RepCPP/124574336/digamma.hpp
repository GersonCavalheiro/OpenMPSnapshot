
#ifndef BOOST_MATH_SF_DIGAMMA_HPP
#define BOOST_MATH_SF_DIGAMMA_HPP

#ifdef _MSC_VER
#pragma once
#pragma warning(push)
#pragma warning(disable:4702) 
#endif

#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/tools/rational.hpp>
#include <boost/math/tools/series.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/mpl/comparison.hpp>
#include <boost/math/tools/big_constant.hpp>

#if defined(__GNUC__) && defined(BOOST_MATH_USE_FLOAT128)
#pragma GCC system_header
#endif

namespace boost{
namespace math{
namespace detail{
inline unsigned digamma_large_lim(const boost::integral_constant<int, 0>*)
{  return 20;  }
inline unsigned digamma_large_lim(const boost::integral_constant<int, 113>*)
{  return 20;  }
inline unsigned digamma_large_lim(const void*)
{  return 10;  }
template <class T>
inline T digamma_imp_large(T x, const boost::integral_constant<int, 113>*)
{
BOOST_MATH_STD_USING 
static const T P[] = {
BOOST_MATH_BIG_CONSTANT(T, 113, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.0083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.003968253968253968253968253968253968253968253968254),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.0041666666666666666666666666666666666666666666666667),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.0075757575757575757575757575757575757575757575757576),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.021092796092796092796092796092796092796092796092796),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.44325980392156862745098039215686274509803921568627),
BOOST_MATH_BIG_CONSTANT(T, 113, 3.0539543302701197438039543302701197438039543302701),
BOOST_MATH_BIG_CONSTANT(T, 113, -26.456212121212121212121212121212121212121212121212),
BOOST_MATH_BIG_CONSTANT(T, 113, 281.4601449275362318840579710144927536231884057971),
BOOST_MATH_BIG_CONSTANT(T, 113, -3607.510546398046398046398046398046398046398046398),
BOOST_MATH_BIG_CONSTANT(T, 113, 54827.583333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 113, -974936.82385057471264367816091954022988505747126437),
BOOST_MATH_BIG_CONSTANT(T, 113, 20052695.796688078946143462272494530559046688078946),
BOOST_MATH_BIG_CONSTANT(T, 113, -472384867.72162990196078431372549019607843137254902),
BOOST_MATH_BIG_CONSTANT(T, 113, 12635724795.916666666666666666666666666666666666667)
};
x -= 1;
T result = log(x);
result += 1 / (2 * x);
T z = 1 / (x*x);
result -= z * tools::evaluate_polynomial(P, z);
return result;
}
template <class T>
inline T digamma_imp_large(T x, const boost::integral_constant<int, 64>*)
{
BOOST_MATH_STD_USING 
static const T P[] = {
BOOST_MATH_BIG_CONSTANT(T, 64, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.0083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.003968253968253968253968253968253968253968253968254),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.0041666666666666666666666666666666666666666666666667),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.0075757575757575757575757575757575757575757575757576),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.021092796092796092796092796092796092796092796092796),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.44325980392156862745098039215686274509803921568627),
BOOST_MATH_BIG_CONSTANT(T, 64, 3.0539543302701197438039543302701197438039543302701),
BOOST_MATH_BIG_CONSTANT(T, 64, -26.456212121212121212121212121212121212121212121212),
BOOST_MATH_BIG_CONSTANT(T, 64, 281.4601449275362318840579710144927536231884057971),
};
x -= 1;
T result = log(x);
result += 1 / (2 * x);
T z = 1 / (x*x);
result -= z * tools::evaluate_polynomial(P, z);
return result;
}
template <class T>
inline T digamma_imp_large(T x, const boost::integral_constant<int, 53>*)
{
BOOST_MATH_STD_USING 
static const T P[] = {
BOOST_MATH_BIG_CONSTANT(T, 53, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.0083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 53, 0.003968253968253968253968253968253968253968253968254),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.0041666666666666666666666666666666666666666666666667),
BOOST_MATH_BIG_CONSTANT(T, 53, 0.0075757575757575757575757575757575757575757575757576),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.021092796092796092796092796092796092796092796092796),
BOOST_MATH_BIG_CONSTANT(T, 53, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.44325980392156862745098039215686274509803921568627)
};
x -= 1;
T result = log(x);
result += 1 / (2 * x);
T z = 1 / (x*x);
result -= z * tools::evaluate_polynomial(P, z);
return result;
}
template <class T>
inline T digamma_imp_large(T x, const boost::integral_constant<int, 24>*)
{
BOOST_MATH_STD_USING 
static const T P[] = {
BOOST_MATH_BIG_CONSTANT(T, 24, 0.083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 24, -0.0083333333333333333333333333333333333333333333333333),
BOOST_MATH_BIG_CONSTANT(T, 24, 0.003968253968253968253968253968253968253968253968254)
};
x -= 1;
T result = log(x);
result += 1 / (2 * x);
T z = 1 / (x*x);
result -= z * tools::evaluate_polynomial(P, z);
return result;
}
template <class T>
struct digamma_series_func
{
private:
int k;
T xx;
T term;
public:
digamma_series_func(T x) : k(1), xx(x * x), term(1 / (x * x)) {}
T operator()()
{
T result = term * boost::math::bernoulli_b2n<T>(k) / (2 * k);
term /= xx;
++k;
return result;
}
typedef T result_type;
};

template <class T, class Policy>
inline T digamma_imp_large(T x, const Policy& pol, const boost::integral_constant<int, 0>*)
{
BOOST_MATH_STD_USING
digamma_series_func<T> s(x);
T result = log(x) - 1 / (2 * x);
boost::uintmax_t max_iter = policies::get_max_series_iterations<Policy>();
result = boost::math::tools::sum_series(s, boost::math::policies::get_epsilon<T, Policy>(), max_iter, -result);
result = -result;
policies::check_series_iterations<T>("boost::math::digamma<%1%>(%1%)", max_iter, pol);
return result;
}
template <class T>
T digamma_imp_1_2(T x, const boost::integral_constant<int, 113>*)
{
static const float Y = 0.99558162689208984375F;

static const T root1 = T(1569415565) / 1073741824uL;
static const T root2 = (T(381566830) / 1073741824uL) / 1073741824uL;
static const T root3 = ((T(111616537) / 1073741824uL) / 1073741824uL) / 1073741824uL;
static const T root4 = (((T(503992070) / 1073741824uL) / 1073741824uL) / 1073741824uL) / 1073741824uL;
static const T root5 = BOOST_MATH_BIG_CONSTANT(T, 113, 0.52112228569249997894452490385577338504019838794544e-36);

static const T P[] = {    
BOOST_MATH_BIG_CONSTANT(T, 113, 0.25479851061131551526977464225335883769),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.18684290534374944114622235683619897417),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.80360876047931768958995775910991929922),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.67227342794829064330498117008564270136),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.26569010991230617151285010695543858005),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.05775672694575986971640757748003553385),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.0071432147823164975485922555833274240665),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.00048740753910766168912364555706064993274),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.16454996865214115723416538844975174761e-4),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.20327832297631728077731148515093164955e-6)
};
static const T Q[] = {    
BOOST_MATH_BIG_CONSTANT(T, 113, 1.0),
BOOST_MATH_BIG_CONSTANT(T, 113, 2.6210924610812025425088411043163287646),
BOOST_MATH_BIG_CONSTANT(T, 113, 2.6850757078559596612621337395886392594),
BOOST_MATH_BIG_CONSTANT(T, 113, 1.4320913706209965531250495490639289418),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.4410872083455009362557012239501953402),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.081385727399251729505165509278152487225),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.0089478633066857163432104815183858149496),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.00055861622855066424871506755481997374154),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.1760168552357342401304462967950178554e-4),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.20585454493572473724556649516040874384e-6),
BOOST_MATH_BIG_CONSTANT(T, 113, -0.90745971844439990284514121823069162795e-11),
BOOST_MATH_BIG_CONSTANT(T, 113, 0.48857673606545846774761343500033283272e-13),
};
T g = x - root1;
g -= root2;
g -= root3;
g -= root4;
g -= root5;
T r = tools::evaluate_polynomial(P, T(x-1)) / tools::evaluate_polynomial(Q, T(x-1));
T result = g * Y + g * r;

return result;
}
template <class T>
T digamma_imp_1_2(T x, const boost::integral_constant<int, 64>*)
{
static const float Y = 0.99558162689208984375F;

static const T root1 = T(1569415565) / 1073741824uL;
static const T root2 = (T(381566830) / 1073741824uL) / 1073741824uL;
static const T root3 = BOOST_MATH_BIG_CONSTANT(T, 64, 0.9016312093258695918615325266959189453125e-19);

static const T P[] = {    
BOOST_MATH_BIG_CONSTANT(T, 64, 0.254798510611315515235),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.314628554532916496608),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.665836341559876230295),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.314767657147375752913),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.0541156266153505273939),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.00289268368333918761452)
};
static const T Q[] = {    
BOOST_MATH_BIG_CONSTANT(T, 64, 1.0),
BOOST_MATH_BIG_CONSTANT(T, 64, 2.1195759927055347547),
BOOST_MATH_BIG_CONSTANT(T, 64, 1.54350554664961128724),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.486986018231042975162),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.0660481487173569812846),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.00298999662592323990972),
BOOST_MATH_BIG_CONSTANT(T, 64, -0.165079794012604905639e-5),
BOOST_MATH_BIG_CONSTANT(T, 64, 0.317940243105952177571e-7)
};
T g = x - root1;
g -= root2;
g -= root3;
T r = tools::evaluate_polynomial(P, T(x-1)) / tools::evaluate_polynomial(Q, T(x-1));
T result = g * Y + g * r;

return result;
}
template <class T>
T digamma_imp_1_2(T x, const boost::integral_constant<int, 53>*)
{
static const float Y = 0.99558162689208984F;

static const T root1 = T(1569415565) / 1073741824uL;
static const T root2 = (T(381566830) / 1073741824uL) / 1073741824uL;
static const T root3 = BOOST_MATH_BIG_CONSTANT(T, 53, 0.9016312093258695918615325266959189453125e-19);

static const T P[] = {    
BOOST_MATH_BIG_CONSTANT(T, 53, 0.25479851061131551),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.32555031186804491),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.65031853770896507),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.28919126444774784),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.045251321448739056),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.0020713321167745952)
};
static const T Q[] = {    
BOOST_MATH_BIG_CONSTANT(T, 53, 1.0),
BOOST_MATH_BIG_CONSTANT(T, 53, 2.0767117023730469),
BOOST_MATH_BIG_CONSTANT(T, 53, 1.4606242909763515),
BOOST_MATH_BIG_CONSTANT(T, 53, 0.43593529692665969),
BOOST_MATH_BIG_CONSTANT(T, 53, 0.054151797245674225),
BOOST_MATH_BIG_CONSTANT(T, 53, 0.0021284987017821144),
BOOST_MATH_BIG_CONSTANT(T, 53, -0.55789841321675513e-6)
};
T g = x - root1;
g -= root2;
g -= root3;
T r = tools::evaluate_polynomial(P, T(x-1)) / tools::evaluate_polynomial(Q, T(x-1));
T result = g * Y + g * r;

return result;
}
template <class T>
inline T digamma_imp_1_2(T x, const boost::integral_constant<int, 24>*)
{
static const float Y = 0.99558162689208984f;
static const T root = 1532632.0f / 1048576;
static const T root_minor = static_cast<T>(0.3700660185912626595423257213284682051735604e-6L);
static const T P[] = {    
0.25479851023250261e0f,
-0.44981331915268368e0f,
-0.43916936919946835e0f,
-0.61041765350579073e-1f
};
static const T Q[] = {    
0.1e1,
0.15890202430554952e1f,
0.65341249856146947e0f,
0.63851690523355715e-1f
};
T g = x - root;
g -= root_minor;
T r = tools::evaluate_polynomial(P, T(x-1)) / tools::evaluate_polynomial(Q, T(x-1));
T result = g * Y + g * r;

return result;
}

template <class T, class Tag, class Policy>
T digamma_imp(T x, const Tag* t, const Policy& pol)
{
BOOST_MATH_STD_USING 

T result = 0;
if(x <= -1)
{
x = 1 - x;
T remainder = x - floor(x);
if(remainder > 0.5)
{
remainder -= 1;
}
if(remainder == 0)
{
return policies::raise_pole_error<T>("boost::math::digamma<%1%>(%1%)", 0, (1-x), pol);
}
result = constants::pi<T>() / tan(constants::pi<T>() * remainder);
}
if(x == 0)
return policies::raise_pole_error<T>("boost::math::digamma<%1%>(%1%)", 0, x, pol);
if(x >= digamma_large_lim(t))
{
result += digamma_imp_large(x, t);
}
else
{
while(x > 2)
{
x -= 1;
result += 1/x;
}
while(x < 1)
{
result -= 1/x;
x += 1;
}
result += digamma_imp_1_2(x, t);
}
return result;
}

template <class T, class Policy>
T digamma_imp(T x, const boost::integral_constant<int, 0>* t, const Policy& pol)
{
BOOST_MATH_STD_USING 

T result = 0;
if(x <= -1)
{
x = 1 - x;
T remainder = x - floor(x);
if(remainder > 0.5)
{
remainder -= 1;
}
if(remainder == 0)
{
return policies::raise_pole_error<T>("boost::math::digamma<%1%>(%1%)", 0, (1 - x), pol);
}
result = constants::pi<T>() / tan(constants::pi<T>() * remainder);
}
if(x == 0)
return policies::raise_pole_error<T>("boost::math::digamma<%1%>(%1%)", 0, x, pol);
int lim = 10 + ((tools::digits<T>() - 50) * 240L) / 950;
T two_x = ldexp(x, 1);
if(x >= lim)
{
result += digamma_imp_large(x, pol, t);
}
else if(floor(x) == x)
{
result = -constants::euler<T, Policy>();
T val = 1;
while(val < x)
{
result += 1 / val;
val += 1;
}
}
else if(floor(two_x) == two_x)
{
result = -2 * constants::ln_two<T, Policy>() - constants::euler<T, Policy>();
int n = itrunc(x);
if(n)
{
for(int k = 1; k < n; ++k)
result += 1 / T(k);
for(int k = n; k <= 2 * n - 1; ++k)
result += 2 / T(k);
}
}
else
{
while(x < lim)
{
result -= 1 / x;
x += 1;
}
result += digamma_imp_large(x, pol, t);
}
return result;
}
template <class T, class Policy>
struct digamma_initializer
{
struct init
{
init()
{
typedef typename policies::precision<T, Policy>::type precision_type;
do_init(boost::integral_constant<bool, precision_type::value && (precision_type::value <= 113)>());
}
void do_init(const boost::true_type&)
{
boost::math::digamma(T(1.5), Policy());
boost::math::digamma(T(500), Policy());
}
void do_init(const false_type&){}
void force_instantiate()const{}
};
static const init initializer;
static void force_instantiate()
{
initializer.force_instantiate();
}
};

template <class T, class Policy>
const typename digamma_initializer<T, Policy>::init digamma_initializer<T, Policy>::initializer;

} 

template <class T, class Policy>
inline typename tools::promote_args<T>::type 
digamma(T x, const Policy&)
{
typedef typename tools::promote_args<T>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::precision<T, Policy>::type precision_type;
typedef boost::integral_constant<int,
(precision_type::value <= 0) || (precision_type::value > 113) ? 0 :
precision_type::value <= 24 ? 24 :
precision_type::value <= 53 ? 53 :
precision_type::value <= 64 ? 64 :
precision_type::value <= 113 ? 113 : 0 > tag_type;
typedef typename policies::normalise<
Policy,
policies::promote_float<false>,
policies::promote_double<false>,
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

detail::digamma_initializer<value_type, forwarding_policy>::force_instantiate();

return policies::checked_narrowing_cast<result_type, Policy>(detail::digamma_imp(
static_cast<value_type>(x),
static_cast<const tag_type*>(0), forwarding_policy()), "boost::math::digamma<%1%>(%1%)");
}

template <class T>
inline typename tools::promote_args<T>::type 
digamma(T x)
{
return digamma(x, policies::policy<>());
}

} 
} 

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#endif

