

#ifndef BOOST_MATH_SP_DETAIL_GAMMA_INVA
#define BOOST_MATH_SP_DETAIL_GAMMA_INVA

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/toms748_solve.hpp>
#include <boost/cstdint.hpp>

namespace boost{ namespace math{ namespace detail{

template <class T, class Policy>
struct gamma_inva_t
{
gamma_inva_t(T z_, T p_, bool invert_) : z(z_), p(p_), invert(invert_) {}
T operator()(T a)
{
return invert ? p - boost::math::gamma_q(a, z, Policy()) : boost::math::gamma_p(a, z, Policy()) - p;
}
private:
T z, p;
bool invert;
};

template <class T, class Policy>
T inverse_poisson_cornish_fisher(T lambda, T p, T q, const Policy& pol)
{
BOOST_MATH_STD_USING
T m = lambda;
T sigma = sqrt(lambda);
T sk = 1 / sigma;
T x = boost::math::erfc_inv(p > q ? 2 * q : 2 * p, pol) * constants::root_two<T>();
if(p < 0.5)
x = -x;
T x2 = x * x;
T w = x + sk * (x2 - 1) / 6;

w = m + sigma * w;
return w > tools::min_value<T>() ? w : tools::min_value<T>();
}

template <class T, class Policy>
T gamma_inva_imp(const T& z, const T& p, const T& q, const Policy& pol)
{
BOOST_MATH_STD_USING  
if(p == 0)
{
return policies::raise_overflow_error<T>("boost::math::gamma_p_inva<%1%>(%1%, %1%)", 0, Policy());
}
if(q == 0)
{
return tools::min_value<T>();
}
gamma_inva_t<T, Policy> f(z, (p < q) ? p : q, (p < q) ? false : true);
tools::eps_tolerance<T> tol(policies::digits<T, Policy>());
T guess;
T factor = 8;
if(z >= 1)
{
guess = 1 + inverse_poisson_cornish_fisher(z, q, p, pol);
if(z > 5)
{
if(z > 1000)
factor = 1.01f;
else if(z > 50)
factor = 1.1f;
else if(guess > 10)
factor = 1.25f;
else
factor = 2;
if(guess < 1.1)
factor = 8;
}
}
else if(z > 0.5)
{
guess = z * 1.2f;
}
else
{
guess = -0.4f / log(z);
}
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
std::pair<T, T> r = bracket_and_solve_root(f, guess, factor, false, tol, max_iter, pol);
if(max_iter >= policies::get_max_root_iterations<Policy>())
return policies::raise_evaluation_error<T>("boost::math::gamma_p_inva<%1%>(%1%, %1%)", "Unable to locate the root within a reasonable number of iterations, closest approximation so far was %1%", r.first, pol);
return (r.first + r.second) / 2;
}

} 

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type 
gamma_p_inva(T1 x, T2 p, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

if(p == 0)
{
policies::raise_overflow_error<result_type>("boost::math::gamma_p_inva<%1%>(%1%, %1%)", 0, Policy());
}
if(p == 1)
{
return tools::min_value<result_type>();
}

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::gamma_inva_imp(
static_cast<value_type>(x), 
static_cast<value_type>(p), 
static_cast<value_type>(1 - static_cast<value_type>(p)), 
pol), "boost::math::gamma_p_inva<%1%>(%1%, %1%)");
}

template <class T1, class T2, class Policy>
inline typename tools::promote_args<T1, T2>::type 
gamma_q_inva(T1 x, T2 q, const Policy& pol)
{
typedef typename tools::promote_args<T1, T2>::type result_type;
typedef typename policies::evaluation<result_type, Policy>::type value_type;
typedef typename policies::normalise<
Policy, 
policies::promote_float<false>, 
policies::promote_double<false>, 
policies::discrete_quantile<>,
policies::assert_undefined<> >::type forwarding_policy;

if(q == 1)
{
policies::raise_overflow_error<result_type>("boost::math::gamma_q_inva<%1%>(%1%, %1%)", 0, Policy());
}
if(q == 0)
{
return tools::min_value<result_type>();
}

return policies::checked_narrowing_cast<result_type, forwarding_policy>(
detail::gamma_inva_imp(
static_cast<value_type>(x), 
static_cast<value_type>(1 - static_cast<value_type>(q)), 
static_cast<value_type>(q), 
pol), "boost::math::gamma_q_inva<%1%>(%1%, %1%)");
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type 
gamma_p_inva(T1 x, T2 p)
{
return boost::math::gamma_p_inva(x, p, policies::policy<>());
}

template <class T1, class T2>
inline typename tools::promote_args<T1, T2>::type
gamma_q_inva(T1 x, T2 q)
{
return boost::math::gamma_q_inva(x, q, policies::policy<>());
}

} 
} 

#endif 



