










#ifndef BOOST_MATH_SPECIAL_NEGATIVE_BINOMIAL_HPP
#define BOOST_MATH_SPECIAL_NEGATIVE_BINOMIAL_HPP

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/special_functions/beta.hpp> 
#include <boost/math/distributions/complement.hpp> 
#include <boost/math/distributions/detail/common_error_handling.hpp> 
#include <boost/math/special_functions/fpclassify.hpp> 
#include <boost/math/tools/roots.hpp> 
#include <boost/math/distributions/detail/inv_discrete_quantile.hpp>

#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/if.hpp>

#include <limits> 
#include <utility>

#if defined (BOOST_MSVC)
#  pragma warning(push)
#endif

namespace boost
{
namespace math
{
namespace negative_binomial_detail
{
template <class RealType, class Policy>
inline bool check_successes(const char* function, const RealType& r, RealType* result, const Policy& pol)
{
if( !(boost::math::isfinite)(r) || (r <= 0) )
{
*result = policies::raise_domain_error<RealType>(
function,
"Number of successes argument is %1%, but must be > 0 !", r, pol);
return false;
}
return true;
}
template <class RealType, class Policy>
inline bool check_success_fraction(const char* function, const RealType& p, RealType* result, const Policy& pol)
{
if( !(boost::math::isfinite)(p) || (p < 0) || (p > 1) )
{
*result = policies::raise_domain_error<RealType>(
function,
"Success fraction argument is %1%, but must be >= 0 and <= 1 !", p, pol);
return false;
}
return true;
}
template <class RealType, class Policy>
inline bool check_dist(const char* function, const RealType& r, const RealType& p, RealType* result, const Policy& pol)
{
return check_success_fraction(function, p, result, pol)
&& check_successes(function, r, result, pol);
}
template <class RealType, class Policy>
inline bool check_dist_and_k(const char* function, const RealType& r, const RealType& p, RealType k, RealType* result, const Policy& pol)
{
if(check_dist(function, r, p, result, pol) == false)
{
return false;
}
if( !(boost::math::isfinite)(k) || (k < 0) )
{ 
*result = policies::raise_domain_error<RealType>(
function,
"Number of failures argument is %1%, but must be >= 0 !", k, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_dist_and_prob(const char* function, const RealType& r, RealType p, RealType prob, RealType* result, const Policy& pol)
{
if((check_dist(function, r, p, result, pol) && detail::check_probability(function, prob, result, pol)) == false)
{
return false;
}
return true;
} 
} 

template <class RealType = double, class Policy = policies::policy<> >
class negative_binomial_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

negative_binomial_distribution(RealType r, RealType p) : m_r(r), m_p(p)
{ 
RealType result;
negative_binomial_detail::check_dist(
"negative_binomial_distribution<%1%>::negative_binomial_distribution",
m_r, 
m_p, 
&result, Policy());
} 

RealType success_fraction() const
{ 
return m_p;
}
RealType successes() const
{ 
return m_r;
}

static RealType find_lower_bound_on_p(
RealType trials,
RealType successes,
RealType alpha) 
{
static const char* function = "boost::math::negative_binomial<%1%>::find_lower_bound_on_p";
RealType result = 0;  
RealType failures = trials - successes;
if(false == detail::check_probability(function, alpha, &result, Policy())
&& negative_binomial_detail::check_dist_and_k(
function, successes, RealType(0), failures, &result, Policy()))
{
return result;
}
return ibeta_inv(successes, failures + 1, alpha, static_cast<RealType*>(0), Policy());
} 

static RealType find_upper_bound_on_p(
RealType trials,
RealType successes,
RealType alpha) 
{
static const char* function = "boost::math::negative_binomial<%1%>::find_upper_bound_on_p";
RealType result = 0;  
RealType failures = trials - successes;
if(false == negative_binomial_detail::check_dist_and_k(
function, successes, RealType(0), failures, &result, Policy())
&& detail::check_probability(function, alpha, &result, Policy()))
{
return result;
}
if(failures == 0)
return 1;
return ibetac_inv(successes, failures, alpha, static_cast<RealType*>(0), Policy());
} 


static RealType find_minimum_number_of_trials(
RealType k,     
RealType p,     
RealType alpha) 
{
static const char* function = "boost::math::negative_binomial<%1%>::find_minimum_number_of_trials";
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_k(
function, RealType(1), p, k, &result, Policy())
&& detail::check_probability(function, alpha, &result, Policy()))
{ return result; }

result = ibeta_inva(k + 1, p, alpha, Policy());  
return result + k;
} 

static RealType find_maximum_number_of_trials(
RealType k,     
RealType p,     
RealType alpha) 
{
static const char* function = "boost::math::negative_binomial<%1%>::find_maximum_number_of_trials";
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_k(
function, RealType(1), p, k, &result, Policy())
&&  detail::check_probability(function, alpha, &result, Policy()))
{ return result; }

result = ibetac_inva(k + 1, p, alpha, Policy());  
return result + k;
} 

private:
RealType m_r; 
RealType m_p; 
}; 

typedef negative_binomial_distribution<double> negative_binomial; 

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const negative_binomial_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>()); 
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const negative_binomial_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0),  max_value<RealType>()); 
}

template <class RealType, class Policy>
inline RealType mean(const negative_binomial_distribution<RealType, Policy>& dist)
{ 
return dist.successes() * (1 - dist.success_fraction() ) / dist.success_fraction();
} 


template <class RealType, class Policy>
inline RealType mode(const negative_binomial_distribution<RealType, Policy>& dist)
{ 
BOOST_MATH_STD_USING 
return floor((dist.successes() -1) * (1 - dist.success_fraction()) / dist.success_fraction());
} 

template <class RealType, class Policy>
inline RealType skewness(const negative_binomial_distribution<RealType, Policy>& dist)
{ 
BOOST_MATH_STD_USING 
RealType p = dist.success_fraction();
RealType r = dist.successes();

return (2 - p) /
sqrt(r * (1 - p));
} 

template <class RealType, class Policy>
inline RealType kurtosis(const negative_binomial_distribution<RealType, Policy>& dist)
{ 
RealType p = dist.success_fraction();
RealType r = dist.successes();
return 3 + (6 / r) + ((p * p) / (r * (1 - p)));
} 

template <class RealType, class Policy>
inline RealType kurtosis_excess(const negative_binomial_distribution<RealType, Policy>& dist)
{ 
RealType p = dist.success_fraction();
RealType r = dist.successes();
return (6 - p * (6-p)) / (r * (1-p));
} 

template <class RealType, class Policy>
inline RealType variance(const negative_binomial_distribution<RealType, Policy>& dist)
{ 
return  dist.successes() * (1 - dist.success_fraction())
/ (dist.success_fraction() * dist.success_fraction());
} 


template <class RealType, class Policy>
inline RealType pdf(const negative_binomial_distribution<RealType, Policy>& dist, const RealType& k)
{ 
BOOST_FPU_EXCEPTION_GUARD

static const char* function = "boost::math::pdf(const negative_binomial_distribution<%1%>&, %1%)";

RealType r = dist.successes();
RealType p = dist.success_fraction();
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_k(
function,
r,
dist.success_fraction(),
k,
&result, Policy()))
{
return result;
}

result = (p/(r + k)) * ibeta_derivative(r, static_cast<RealType>(k+1), p, Policy());
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const negative_binomial_distribution<RealType, Policy>& dist, const RealType& k)
{ 
static const char* function = "boost::math::cdf(const negative_binomial_distribution<%1%>&, %1%)";
using boost::math::ibeta; 
RealType p = dist.success_fraction();
RealType r = dist.successes();
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_k(
function,
r,
dist.success_fraction(),
k,
&result, Policy()))
{
return result;
}

RealType probability = ibeta(r, static_cast<RealType>(k+1), p, Policy());
return probability;
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<negative_binomial_distribution<RealType, Policy>, RealType>& c)
{ 

static const char* function = "boost::math::cdf(const negative_binomial_distribution<%1%>&, %1%)";
using boost::math::ibetac; 
RealType const& k = c.param;
negative_binomial_distribution<RealType, Policy> const& dist = c.dist;
RealType p = dist.success_fraction();
RealType r = dist.successes();
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_k(
function,
r,
p,
k,
&result, Policy()))
{
return result;
}
RealType probability = ibetac(r, static_cast<RealType>(k+1), p, Policy());
return probability;
} 

template <class RealType, class Policy>
inline RealType quantile(const negative_binomial_distribution<RealType, Policy>& dist, const RealType& P)
{ 

static const char* function = "boost::math::quantile(const negative_binomial_distribution<%1%>&, %1%)";
BOOST_MATH_STD_USING 

RealType p = dist.success_fraction();
RealType r = dist.successes();
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_prob
(function, r, p, P, &result, Policy()))
{
return result;
}

if (P == 1)
{  
result = policies::raise_overflow_error<RealType>(
function,
"Probability argument is 1, which implies infinite failures !", Policy());
return result;
}
if (P == 0)
{ 
return 0; 
}
if (P <= pow(dist.success_fraction(), dist.successes()))
{ 
return 0;
}
if(p == 0)
{  
result = policies::raise_overflow_error<RealType>(
function,
"Success fraction is 0, which implies infinite failures !", Policy());
return result;
}

RealType guess = 0;
RealType factor = 5;
if(r * r * r * P * p > 0.005)
guess = detail::inverse_negative_binomial_cornish_fisher(r, p, RealType(1-p), P, RealType(1-P), Policy());

if(guess < 10)
{
guess = (std::min)(RealType(r * 2), RealType(10));
}
else
factor = (1-P < sqrt(tools::epsilon<RealType>())) ? 2 : (guess < 20 ? 1.2f : 1.1f);
BOOST_MATH_INSTRUMENT_CODE("guess = " << guess);
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
typedef typename Policy::discrete_quantile_type discrete_type;
return detail::inverse_discrete_quantile(
dist,
P,
false,
guess,
factor,
RealType(1),
discrete_type(),
max_iter);
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<negative_binomial_distribution<RealType, Policy>, RealType>& c)
{  
static const char* function = "boost::math::quantile(const negative_binomial_distribution<%1%>&, %1%)";
BOOST_MATH_STD_USING

RealType Q = c.param;
const negative_binomial_distribution<RealType, Policy>& dist = c.dist;
RealType p = dist.success_fraction();
RealType r = dist.successes();
RealType result = 0;
if(false == negative_binomial_detail::check_dist_and_prob(
function,
r,
p,
Q,
&result, Policy()))
{
return result;
}

if(Q == 1)
{  
return 0; 
}
if(Q == 0)
{  
result = policies::raise_overflow_error<RealType>(
function,
"Probability argument complement is 0, which implies infinite failures !", Policy());
return result;
}
if (-Q <= boost::math::powm1(dist.success_fraction(), dist.successes(), Policy()))
{  
return 0; 
}
if(p == 0)
{  
result = policies::raise_overflow_error<RealType>(
function,
"Success fraction is 0, which implies infinite failures !", Policy());
return result;
}
RealType guess = 0;
RealType factor = 5;
if(r * r * r * (1-Q) * p > 0.005)
guess = detail::inverse_negative_binomial_cornish_fisher(r, p, RealType(1-p), RealType(1-Q), Q, Policy());

if(guess < 10)
{
guess = (std::min)(RealType(r * 2), RealType(10));
}
else
factor = (Q < sqrt(tools::epsilon<RealType>())) ? 2 : (guess < 20 ? 1.2f : 1.1f);
BOOST_MATH_INSTRUMENT_CODE("guess = " << guess);
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
typedef typename Policy::discrete_quantile_type discrete_type;
return detail::inverse_discrete_quantile(
dist,
Q,
true,
guess,
factor,
RealType(1),
discrete_type(),
max_iter);
} 

} 
} 

#include <boost/math/distributions/detail/derived_accessors.hpp>

#if defined (BOOST_MSVC)
# pragma warning(pop)
#endif

#endif 
