

#ifndef BOOST_STATS_CAUCHY_HPP
#define BOOST_STATS_CAUCHY_HPP

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127) 
#endif

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/config/no_tr1/cmath.hpp>

#include <utility>

namespace boost{ namespace math
{

template <class RealType, class Policy>
class cauchy_distribution;

namespace detail
{

template <class RealType, class Policy>
RealType cdf_imp(const cauchy_distribution<RealType, Policy>& dist, const RealType& x, bool complement)
{
BOOST_MATH_STD_USING 
static const char* function = "boost::math::cdf(cauchy<%1%>&, %1%)";
RealType result = 0;
RealType location = dist.location();
RealType scale = dist.scale();
if(false == detail::check_location(function, location, &result, Policy()))
{
return result;
}
if(false == detail::check_scale(function, scale, &result, Policy()))
{
return result;
}
if(std::numeric_limits<RealType>::has_infinity && x == std::numeric_limits<RealType>::infinity())
{ 
return static_cast<RealType>((complement) ? 0 : 1);
}
if(std::numeric_limits<RealType>::has_infinity && x == -std::numeric_limits<RealType>::infinity())
{ 
return static_cast<RealType>((complement) ? 1 : 0);
}
if(false == detail::check_x(function, x, &result, Policy()))
{ 
return result;
}
RealType mx = -fabs((x - location) / scale); 
if(mx > -tools::epsilon<RealType>() / 8)
{  
return 0.5;
}
result = -atan(1 / mx) / constants::pi<RealType>();
return (((x > location) != complement) ? 1 - result : result);
} 

template <class RealType, class Policy>
RealType quantile_imp(
const cauchy_distribution<RealType, Policy>& dist,
const RealType& p,
bool complement)
{
static const char* function = "boost::math::quantile(cauchy<%1%>&, %1%)";
BOOST_MATH_STD_USING 

RealType result = 0;
RealType location = dist.location();
RealType scale = dist.scale();
if(false == detail::check_location(function, location, &result, Policy()))
{
return result;
}
if(false == detail::check_scale(function, scale, &result, Policy()))
{
return result;
}
if(false == detail::check_probability(function, p, &result, Policy()))
{
return result;
}
if(p == 1)
{
return (complement ? -1 : 1) * policies::raise_overflow_error<RealType>(function, 0, Policy());
}
if(p == 0)
{
return (complement ? 1 : -1) * policies::raise_overflow_error<RealType>(function, 0, Policy());
}

RealType P = p - floor(p);   
if(P > 0.5)
{
P = P - 1;
}
if(P == 0.5)   
{
return location;
}
result = -scale / tan(constants::pi<RealType>() * P);
return complement ? RealType(location - result) : RealType(location + result);
} 

} 

template <class RealType = double, class Policy = policies::policy<> >
class cauchy_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

cauchy_distribution(RealType l_location = 0, RealType l_scale = 1)
: m_a(l_location), m_hg(l_scale)
{
static const char* function = "boost::math::cauchy_distribution<%1%>::cauchy_distribution";
RealType result;
detail::check_location(function, l_location, &result, Policy());
detail::check_scale(function, l_scale, &result, Policy());
} 

RealType location()const
{
return m_a;
}
RealType scale()const
{
return m_hg;
}

private:
RealType m_a;    
RealType m_hg;   
};

typedef cauchy_distribution<double> cauchy;

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const cauchy_distribution<RealType, Policy>&)
{ 
if (std::numeric_limits<RealType>::has_infinity)
{ 
return std::pair<RealType, RealType>(-std::numeric_limits<RealType>::infinity(), std::numeric_limits<RealType>::infinity()); 
}
else
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(-max_value<RealType>(), max_value<RealType>()); 
}
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const cauchy_distribution<RealType, Policy>& )
{ 
if (std::numeric_limits<RealType>::has_infinity)
{ 
return std::pair<RealType, RealType>(-std::numeric_limits<RealType>::infinity(), std::numeric_limits<RealType>::infinity()); 
}
else
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(-tools::max_value<RealType>(), max_value<RealType>()); 
}
}

template <class RealType, class Policy>
inline RealType pdf(const cauchy_distribution<RealType, Policy>& dist, const RealType& x)
{  
BOOST_MATH_STD_USING  

static const char* function = "boost::math::pdf(cauchy<%1%>&, %1%)";
RealType result = 0;
RealType location = dist.location();
RealType scale = dist.scale();
if(false == detail::check_scale("boost::math::pdf(cauchy<%1%>&, %1%)", scale, &result, Policy()))
{
return result;
}
if(false == detail::check_location("boost::math::pdf(cauchy<%1%>&, %1%)", location, &result, Policy()))
{
return result;
}
if((boost::math::isinf)(x))
{
return 0; 
}

if(false == detail::check_x(function, x, &result, Policy()))
{ 
return result;
}

RealType xs = (x - location) / scale;
result = 1 / (constants::pi<RealType>() * scale * (1 + xs * xs));
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const cauchy_distribution<RealType, Policy>& dist, const RealType& x)
{
return detail::cdf_imp(dist, x, false);
} 

template <class RealType, class Policy>
inline RealType quantile(const cauchy_distribution<RealType, Policy>& dist, const RealType& p)
{
return detail::quantile_imp(dist, p, false);
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<cauchy_distribution<RealType, Policy>, RealType>& c)
{
return detail::cdf_imp(c.dist, c.param, true);
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<cauchy_distribution<RealType, Policy>, RealType>& c)
{
return detail::quantile_imp(c.dist, c.param, true);
} 

template <class RealType, class Policy>
inline RealType mean(const cauchy_distribution<RealType, Policy>&)
{  
typedef typename Policy::assert_undefined_type assert_type;
BOOST_STATIC_ASSERT(assert_type::value == 0);

return policies::raise_domain_error<RealType>(
"boost::math::mean(cauchy<%1%>&)",
"The Cauchy distribution does not have a mean: "
"the only possible return value is %1%.",
std::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
inline RealType variance(const cauchy_distribution<RealType, Policy>& )
{
typedef typename Policy::assert_undefined_type assert_type;
BOOST_STATIC_ASSERT(assert_type::value == 0);

return policies::raise_domain_error<RealType>(
"boost::math::variance(cauchy<%1%>&)",
"The Cauchy distribution does not have a variance: "
"the only possible return value is %1%.",
std::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
inline RealType mode(const cauchy_distribution<RealType, Policy>& dist)
{
return dist.location();
}

template <class RealType, class Policy>
inline RealType median(const cauchy_distribution<RealType, Policy>& dist)
{
return dist.location();
}
template <class RealType, class Policy>
inline RealType skewness(const cauchy_distribution<RealType, Policy>& )
{
typedef typename Policy::assert_undefined_type assert_type;
BOOST_STATIC_ASSERT(assert_type::value == 0);

return policies::raise_domain_error<RealType>(
"boost::math::skewness(cauchy<%1%>&)",
"The Cauchy distribution does not have a skewness: "
"the only possible return value is %1%.",
std::numeric_limits<RealType>::quiet_NaN(), Policy()); 
}

template <class RealType, class Policy>
inline RealType kurtosis(const cauchy_distribution<RealType, Policy>& )
{
typedef typename Policy::assert_undefined_type assert_type;
BOOST_STATIC_ASSERT(assert_type::value == 0);

return policies::raise_domain_error<RealType>(
"boost::math::kurtosis(cauchy<%1%>&)",
"The Cauchy distribution does not have a kurtosis: "
"the only possible return value is %1%.",
std::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const cauchy_distribution<RealType, Policy>& )
{
typedef typename Policy::assert_undefined_type assert_type;
BOOST_STATIC_ASSERT(assert_type::value == 0);

return policies::raise_domain_error<RealType>(
"boost::math::kurtosis_excess(cauchy<%1%>&)",
"The Cauchy distribution does not have a kurtosis: "
"the only possible return value is %1%.",
std::numeric_limits<RealType>::quiet_NaN(), Policy());
}

template <class RealType, class Policy>
inline RealType entropy(const cauchy_distribution<RealType, Policy> & dist)
{
using std::log;
return log(2*constants::two_pi<RealType>()*dist.scale());
}

} 
} 

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif 
