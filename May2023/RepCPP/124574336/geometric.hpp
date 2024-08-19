







#ifndef BOOST_MATH_SPECIAL_GEOMETRIC_HPP
#define BOOST_MATH_SPECIAL_GEOMETRIC_HPP

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
namespace geometric_detail
{
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
inline bool check_dist(const char* function, const RealType& p, RealType* result, const Policy& pol)
{
return check_success_fraction(function, p, result, pol);
}

template <class RealType, class Policy>
inline bool check_dist_and_k(const char* function,  const RealType& p, RealType k, RealType* result, const Policy& pol)
{
if(check_dist(function, p, result, pol) == false)
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
inline bool check_dist_and_prob(const char* function, RealType p, RealType prob, RealType* result, const Policy& pol)
{
if((check_dist(function, p, result, pol) && detail::check_probability(function, prob, result, pol)) == false)
{
return false;
}
return true;
} 
} 

template <class RealType = double, class Policy = policies::policy<> >
class geometric_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

geometric_distribution(RealType p) : m_p(p)
{ 
RealType result;
geometric_detail::check_dist(
"geometric_distribution<%1%>::geometric_distribution",
m_p, 
&result, Policy());
} 

RealType success_fraction() const
{ 
return m_p;
}
RealType successes() const
{ 
return 1;
}

static RealType find_lower_bound_on_p(
RealType trials,
RealType alpha) 
{
static const char* function = "boost::math::geometric<%1%>::find_lower_bound_on_p";
RealType result = 0;  
RealType successes = 1;
RealType failures = trials - successes;
if(false == detail::check_probability(function, alpha, &result, Policy())
&& geometric_detail::check_dist_and_k(
function, RealType(0), failures, &result, Policy()))
{
return result;
}
return ibeta_inv(successes, failures + 1, alpha, static_cast<RealType*>(0), Policy());
} 

static RealType find_upper_bound_on_p(
RealType trials,
RealType alpha) 
{
static const char* function = "boost::math::geometric<%1%>::find_upper_bound_on_p";
RealType result = 0;  
RealType successes = 1;
RealType failures = trials - successes;
if(false == geometric_detail::check_dist_and_k(
function, RealType(0), failures, &result, Policy())
&& detail::check_probability(function, alpha, &result, Policy()))
{
return result;
}
if(failures == 0)
{
return 1;
}
return ibetac_inv(successes, failures, alpha, static_cast<RealType*>(0), Policy());
} 


static RealType find_minimum_number_of_trials(
RealType k,     
RealType p,     
RealType alpha) 
{
static const char* function = "boost::math::geometric<%1%>::find_minimum_number_of_trials";
RealType result = 0;
if(false == geometric_detail::check_dist_and_k(
function, p, k, &result, Policy())
&& detail::check_probability(function, alpha, &result, Policy()))
{
return result;
}
result = ibeta_inva(k + 1, p, alpha, Policy());  
return result + k;
} 

static RealType find_maximum_number_of_trials(
RealType k,     
RealType p,     
RealType alpha) 
{
static const char* function = "boost::math::geometric<%1%>::find_maximum_number_of_trials";
RealType result = 0;
if(false == geometric_detail::check_dist_and_k(
function, p, k, &result, Policy())
&&  detail::check_probability(function, alpha, &result, Policy()))
{ 
return result;
}
result = ibetac_inva(k + 1, p, alpha, Policy());  
return result + k;
} 

private:
RealType m_p; 
}; 

typedef geometric_distribution<double> geometric; 

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const geometric_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>()); 
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const geometric_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0),  max_value<RealType>()); 
}

template <class RealType, class Policy>
inline RealType mean(const geometric_distribution<RealType, Policy>& dist)
{ 
return (1 - dist.success_fraction() ) / dist.success_fraction();
} 


template <class RealType, class Policy>
inline RealType mode(const geometric_distribution<RealType, Policy>&)
{ 
BOOST_MATH_STD_USING 
return 0;
} 

template <class RealType, class Policy>
inline RealType variance(const geometric_distribution<RealType, Policy>& dist)
{ 
return  (1 - dist.success_fraction())
/ (dist.success_fraction() * dist.success_fraction());
} 

template <class RealType, class Policy>
inline RealType skewness(const geometric_distribution<RealType, Policy>& dist)
{ 
BOOST_MATH_STD_USING 
RealType p = dist.success_fraction();
return (2 - p) / sqrt(1 - p);
} 

template <class RealType, class Policy>
inline RealType kurtosis(const geometric_distribution<RealType, Policy>& dist)
{ 
RealType p = dist.success_fraction();
return 3 + (p*p - 6*p + 6) / (1 - p);
} 

template <class RealType, class Policy>
inline RealType kurtosis_excess(const geometric_distribution<RealType, Policy>& dist)
{ 
RealType p = dist.success_fraction();
return (p*p - 6*p + 6) / (1 - p);
} 


template <class RealType, class Policy>
inline RealType pdf(const geometric_distribution<RealType, Policy>& dist, const RealType& k)
{ 
BOOST_FPU_EXCEPTION_GUARD
BOOST_MATH_STD_USING  
static const char* function = "boost::math::pdf(const geometric_distribution<%1%>&, %1%)";

RealType p = dist.success_fraction();
RealType result = 0;
if(false == geometric_detail::check_dist_and_k(
function,
p,
k,
&result, Policy()))
{
return result;
}
if (k == 0)
{
return p; 
}
RealType q = 1 - p;  
result = p * pow(q, k);
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const geometric_distribution<RealType, Policy>& dist, const RealType& k)
{ 
static const char* function = "boost::math::cdf(const geometric_distribution<%1%>&, %1%)";

RealType p = dist.success_fraction();
RealType result = 0;
if(false == geometric_detail::check_dist_and_k(
function,
p,
k,
&result, Policy()))
{
return result;
}
if(k == 0)
{
return p; 
}

RealType z = boost::math::log1p(-p, Policy()) * (k + 1);
RealType probability = -boost::math::expm1(z, Policy());

return probability;
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<geometric_distribution<RealType, Policy>, RealType>& c)
{ 
BOOST_MATH_STD_USING
static const char* function = "boost::math::cdf(const geometric_distribution<%1%>&, %1%)";
RealType const& k = c.param;
geometric_distribution<RealType, Policy> const& dist = c.dist;
RealType p = dist.success_fraction();
RealType result = 0;
if(false == geometric_detail::check_dist_and_k(
function,
p,
k,
&result, Policy()))
{
return result;
}
RealType z = boost::math::log1p(-p, Policy()) * (k+1);
RealType probability = exp(z);
return probability;
} 

template <class RealType, class Policy>
inline RealType quantile(const geometric_distribution<RealType, Policy>& dist, const RealType& x)
{ 


static const char* function = "boost::math::quantile(const geometric_distribution<%1%>&, %1%)";
BOOST_MATH_STD_USING 

RealType success_fraction = dist.success_fraction();
RealType result = 0;
if(false == geometric_detail::check_dist_and_prob
(function, success_fraction, x, &result, Policy()))
{
return result;
}

if (x == 1)
{  
result = policies::raise_overflow_error<RealType>(
function,
"Probability argument is 1, which implies infinite failures !", Policy());
return result;
}
if (x == 0)
{ 
return 0; 
}
if (x <= success_fraction)
{ 
return 0;
}
if (x == 1)
{
return 0;
}

result = boost::math::log1p(-x, Policy()) / boost::math::log1p(-success_fraction, Policy()) - 1;
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<geometric_distribution<RealType, Policy>, RealType>& c)
{  
static const char* function = "boost::math::quantile(const geometric_distribution<%1%>&, %1%)";
BOOST_MATH_STD_USING
RealType x = c.param;
const geometric_distribution<RealType, Policy>& dist = c.dist;
RealType success_fraction = dist.success_fraction();
RealType result = 0;
if(false == geometric_detail::check_dist_and_prob(
function,
success_fraction,
x,
&result, Policy()))
{
return result;
}

if(x == 1)
{  
return 0; 
}
if (-x <= boost::math::powm1(dist.success_fraction(), dist.successes(), Policy()))
{  
return 0; 
}
if(x == 0)
{  
result = policies::raise_overflow_error<RealType>(
function,
"Probability argument complement is 0, which implies infinite failures !", Policy());
return result;
}
result = log(x) / boost::math::log1p(-success_fraction, Policy()) - 1;
return result;

} 

} 
} 

#include <boost/math/distributions/detail/derived_accessors.hpp>

#if defined (BOOST_MSVC)
# pragma warning(pop)
#endif

#endif 
