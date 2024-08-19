

#ifndef BOOST_MATH_SP_DETAIL_BETA_INV_AB
#define BOOST_MATH_SP_DETAIL_BETA_INV_AB

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/toms748_solve.hpp>
#include <boost/cstdint.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
struct beta_inv_ab_t
{
beta_inv_ab_t(T b_, T z_, T p_, bool invert_, bool swap_ab_) : b(b_), z(z_), p(p_), invert(invert_), swap_ab(swap_ab_) {}
T operator()(T a)
{
return invert ? 
p - boost::math::ibetac(swap_ab ? b : a, swap_ab ? a : b, z, Policy()) 
: boost::math::ibeta(swap_ab ? b : a, swap_ab ? a : b, z, Policy()) - p;
}
private:
T b, z, p;
bool invert, swap_ab;
};

template <class T, class Policy>
T inverse_negative_binomial_cornish_fisher(T n, T sf, T sfc, T p, T q, const Policy& pol)
{
BOOST_MATH_STD_USING
T m = n * (sfc) / sf;
T t = sqrt(n * (sfc));
T sigma = t / sf;
T sk = (1 + sfc) / t;
T k = (6 - sf * (5+sfc)) / (n * (sfc));
T x = boost::math::erfc_inv(p > q ? 2 * q : 2 * p, pol) * constants::root_two<T>();
if(p < 0.5)
x = -x;
T x2 = x * x;
T w = x + sk * (x2 - 1) / 6;
if(n >= 10)
w += k * x * (x2 - 3) / 24 + sk * sk * x * (2 * x2 - 5) / -36;

w = m + sigma * w;
if(w < tools::min_value<T>())
return tools::min_value<T>();
return w;
}

template <class T, class Policy>
T ibeta_inv_ab_imp(const T& b, const T& z, const T& p, const T& q, bool swap_ab, const Policy& pol)
{
BOOST_MATH_STD_USING  
BOOST_MATH_INSTRUMENT_CODE("b = " << b << " z = " << z << " p = " << p << " q = " << " swap = " << swap_ab);
if(p == 0)
{
return swap_ab ? tools::min_value<T>() : tools::max_value<T>();
}
if(q == 0)
{
return swap_ab ? tools::max_value<T>() : tools::min_value<T>();
}
beta_inv_ab_t<T, Policy> f(b, z, (p < q) ? p : q, (p < q) ? false : true, swap_ab);
tools::eps_tolerance<T> tol(policies::digits<T, Policy>());
T guess = 0;
T factor = 5;
T n = b;
T sf = swap_ab ? z : 1-z;
T sfc = swap_ab ? 1-z : z;
T u = swap_ab ? p : q;
T v = swap_ab ? q : p;
if(u <= pow(sf, n))
{
if((p < q) != swap_ab)
{
guess = (std::min)(T(b * 2), T(1));
}
else
{
guess = (std::min)(T(b / 2), T(1));
}
}
if(n * n * n * u * sf > 0.005)
guess = 1 + inverse_negative_binomial_cornish_fisher(n, sf, sfc, u, v, pol);

if(guess < 10)
{
if((p < q) != swap_ab)
{
guess = (std::min)(T(b * 2), T(10));
}
else
{
guess = (std::min)(T(b / 2), T(10));
}
}
else
factor = (v < sqrt(tools::epsilon<T>())) ? 2 : (guess < 20 ? 1.2f : 1.1f);
BOOST_MATH_INSTRUMENT_CODE("guess = " << guess);
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
std::pair<T, T> r = bracket_and_solve_root(f, guess, factor, swap_ab ? true : false, tol, max_iter, pol);
if(max_iter >= policies::get_max_root_iterations<Policy>())
return policies::raise_evaluation_error<T>("boost::math::ibeta_invab_imp<%1%>(%1%,%1%,%1%)", "Unable to locate the root within a reasonable number of iterations, closest approximation so far was %1%", r.first, pol);
return (r.first + r.second) / 2;
}

} 

template <class RT1, class RT2, class RT3, class Policy>
typename tools::promote_args<RT1, RT2, RT3>::type 
ibeta_inva(RT1 b, RT2 x, RT3 p, const Policy& pol)
{
typedef typename tools::promote_args<RT1, RT2, RT3>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

static const char* function = "boost::math::ibeta_inva<%1%>(%1%,%1%,%1%)";
if(p == 0)
{
return policies::raise_overflow_error<result_type>(function, 0, Policy());
}
if(p == 1)
{
return tools::min_value<result_type>();
}

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::ibeta_inv_ab_imp(
static_cast<value_type>(b), 
static_cast<value_type>(x), 
static_cast<value_type>(p), 
static_cast<value_type>(1 - static_cast<value_type>(p)), 
false, pol), 
function);
}

template <class RT1, class RT2, class RT3, class Policy>
typename tools::promote_args<RT1, RT2, RT3>::type 
ibetac_inva(RT1 b, RT2 x, RT3 q, const Policy& pol)
{
typedef typename tools::promote_args<RT1, RT2, RT3>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

static const char* function = "boost::math::ibetac_inva<%1%>(%1%,%1%,%1%)";
if(q == 1)
{
return policies::raise_overflow_error<result_type>(function, 0, Policy());
}
if(q == 0)
{
return tools::min_value<result_type>();
}

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::ibeta_inv_ab_imp(
static_cast<value_type>(b), 
static_cast<value_type>(x), 
static_cast<value_type>(1 - static_cast<value_type>(q)), 
static_cast<value_type>(q), 
false, pol),
function);
}

template <class RT1, class RT2, class RT3, class Policy>
typename tools::promote_args<RT1, RT2, RT3>::type 
ibeta_invb(RT1 a, RT2 x, RT3 p, const Policy& pol)
{
typedef typename tools::promote_args<RT1, RT2, RT3>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

static const char* function = "boost::math::ibeta_invb<%1%>(%1%,%1%,%1%)";
if(p == 0)
{
return tools::min_value<result_type>();
}
if(p == 1)
{
return policies::raise_overflow_error<result_type>(function, 0, Policy());
}

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::ibeta_inv_ab_imp(
static_cast<value_type>(a), 
static_cast<value_type>(x), 
static_cast<value_type>(p), 
static_cast<value_type>(1 - static_cast<value_type>(p)), 
true, pol),
function);
}

template <class RT1, class RT2, class RT3, class Policy>
typename tools::promote_args<RT1, RT2, RT3>::type 
ibetac_invb(RT1 a, RT2 x, RT3 q, const Policy& pol)
{
static const char* function = "boost::math::ibeta_invb<%1%>(%1%, %1%, %1%)";
typedef typename tools::promote_args<RT1, RT2, RT3>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

if(q == 1)
{
return tools::min_value<result_type>();
}
if(q == 0)
{
return policies::raise_overflow_error<result_type>(function, 0, Policy());
}

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::ibeta_inv_ab_imp(
static_cast<value_type>(a), 
static_cast<value_type>(x), 
static_cast<value_type>(1 - static_cast<value_type>(q)), 
static_cast<value_type>(q),
true, pol),
function);
}

template <class RT1, class RT2, class RT3>
inline typename tools::promote_args<RT1, RT2, RT3>::type 
ibeta_inva(RT1 b, RT2 x, RT3 p)
{
return boost::math::ibeta_inva(b, x, p, policies::policy<>());
}

template <class RT1, class RT2, class RT3>
inline typename tools::promote_args<RT1, RT2, RT3>::type 
ibetac_inva(RT1 b, RT2 x, RT3 q)
{
return boost::math::ibetac_inva(b, x, q, policies::policy<>());
}

template <class RT1, class RT2, class RT3>
inline typename tools::promote_args<RT1, RT2, RT3>::type 
ibeta_invb(RT1 a, RT2 x, RT3 p)
{
return boost::math::ibeta_invb(a, x, p, policies::policy<>());
}

template <class RT1, class RT2, class RT3>
inline typename tools::promote_args<RT1, RT2, RT3>::type 
ibetac_invb(RT1 a, RT2 x, RT3 q)
{
return boost::math::ibetac_invb(a, x, q, policies::policy<>());
}

} 
} 

#endif 



