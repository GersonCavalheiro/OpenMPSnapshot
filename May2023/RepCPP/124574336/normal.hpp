

#ifndef BOOST_STATS_NORMAL_HPP
#define BOOST_STATS_NORMAL_HPP


#include <boost/math/distributions/fwd.hpp>
#include <boost/math/special_functions/erf.hpp> 
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

#include <utility>

namespace boost{ namespace math{

template <class RealType = double, class Policy = policies::policy<> >
class normal_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

normal_distribution(RealType l_mean = 0, RealType sd = 1)
: m_mean(l_mean), m_sd(sd)
{ 
static const char* function = "boost::math::normal_distribution<%1%>::normal_distribution";

RealType result;
detail::check_scale(function, sd, &result, Policy());
detail::check_location(function, l_mean, &result, Policy());
}

RealType mean()const
{ 
return m_mean;
}

RealType standard_deviation()const
{ 
return m_sd;
}

RealType location()const
{ 
return m_mean;
}
RealType scale()const
{ 
return m_sd;
}

private:
RealType m_mean;  
RealType m_sd;    
}; 

typedef normal_distribution<double> normal;

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)
#endif

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const normal_distribution<RealType, Policy>& )
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
inline const std::pair<RealType, RealType> support(const normal_distribution<RealType, Policy>& )
{ 
if (std::numeric_limits<RealType>::has_infinity)
{ 
return std::pair<RealType, RealType>(-std::numeric_limits<RealType>::infinity(), std::numeric_limits<RealType>::infinity()); 
}
else
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(-max_value<RealType>(),  max_value<RealType>()); 
}
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class RealType, class Policy>
inline RealType pdf(const normal_distribution<RealType, Policy>& dist, const RealType& x)
{
BOOST_MATH_STD_USING  

RealType sd = dist.standard_deviation();
RealType mean = dist.mean();

static const char* function = "boost::math::pdf(const normal_distribution<%1%>&, %1%)";

RealType result = 0;
if(false == detail::check_scale(function, sd, &result, Policy()))
{
return result;
}
if(false == detail::check_location(function, mean, &result, Policy()))
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

RealType exponent = x - mean;
exponent *= -exponent;
exponent /= 2 * sd * sd;

result = exp(exponent);
result /= sd * sqrt(2 * constants::pi<RealType>());

return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const normal_distribution<RealType, Policy>& dist, const RealType& x)
{
BOOST_MATH_STD_USING  

RealType sd = dist.standard_deviation();
RealType mean = dist.mean();
static const char* function = "boost::math::cdf(const normal_distribution<%1%>&, %1%)";
RealType result = 0;
if(false == detail::check_scale(function, sd, &result, Policy()))
{
return result;
}
if(false == detail::check_location(function, mean, &result, Policy()))
{
return result;
}
if((boost::math::isinf)(x))
{
if(x < 0) return 0; 
return 1; 
}
if(false == detail::check_x(function, x, &result, Policy()))
{
return result;
}
RealType diff = (x - mean) / (sd * constants::root_two<RealType>());
result = boost::math::erfc(-diff, Policy()) / 2;
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const normal_distribution<RealType, Policy>& dist, const RealType& p)
{
BOOST_MATH_STD_USING  

RealType sd = dist.standard_deviation();
RealType mean = dist.mean();
static const char* function = "boost::math::quantile(const normal_distribution<%1%>&, %1%)";

RealType result = 0;
if(false == detail::check_scale(function, sd, &result, Policy()))
return result;
if(false == detail::check_location(function, mean, &result, Policy()))
return result;
if(false == detail::check_probability(function, p, &result, Policy()))
return result;

result= boost::math::erfc_inv(2 * p, Policy());
result = -result;
result *= sd * constants::root_two<RealType>();
result += mean;
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<normal_distribution<RealType, Policy>, RealType>& c)
{
BOOST_MATH_STD_USING  

RealType sd = c.dist.standard_deviation();
RealType mean = c.dist.mean();
RealType x = c.param;
static const char* function = "boost::math::cdf(const complement(normal_distribution<%1%>&), %1%)";

RealType result = 0;
if(false == detail::check_scale(function, sd, &result, Policy()))
return result;
if(false == detail::check_location(function, mean, &result, Policy()))
return result;
if((boost::math::isinf)(x))
{
if(x < 0) return 1; 
return 0; 
}
if(false == detail::check_x(function, x, &result, Policy()))
return result;

RealType diff = (x - mean) / (sd * constants::root_two<RealType>());
result = boost::math::erfc(diff, Policy()) / 2;
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<normal_distribution<RealType, Policy>, RealType>& c)
{
BOOST_MATH_STD_USING  

RealType sd = c.dist.standard_deviation();
RealType mean = c.dist.mean();
static const char* function = "boost::math::quantile(const complement(normal_distribution<%1%>&), %1%)";
RealType result = 0;
if(false == detail::check_scale(function, sd, &result, Policy()))
return result;
if(false == detail::check_location(function, mean, &result, Policy()))
return result;
RealType q = c.param;
if(false == detail::check_probability(function, q, &result, Policy()))
return result;
result = boost::math::erfc_inv(2 * q, Policy());
result *= sd * constants::root_two<RealType>();
result += mean;
return result;
} 

template <class RealType, class Policy>
inline RealType mean(const normal_distribution<RealType, Policy>& dist)
{
return dist.mean();
}

template <class RealType, class Policy>
inline RealType standard_deviation(const normal_distribution<RealType, Policy>& dist)
{
return dist.standard_deviation();
}

template <class RealType, class Policy>
inline RealType mode(const normal_distribution<RealType, Policy>& dist)
{
return dist.mean();
}

template <class RealType, class Policy>
inline RealType median(const normal_distribution<RealType, Policy>& dist)
{
return dist.mean();
}

template <class RealType, class Policy>
inline RealType skewness(const normal_distribution<RealType, Policy>& )
{
return 0;
}

template <class RealType, class Policy>
inline RealType kurtosis(const normal_distribution<RealType, Policy>& )
{
return 3;
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const normal_distribution<RealType, Policy>& )
{
return 0;
}

template <class RealType, class Policy>
inline RealType entropy(const normal_distribution<RealType, Policy> & dist)
{
using std::log;
RealType arg = constants::two_pi<RealType>()*constants::e<RealType>()*dist.standard_deviation()*dist.standard_deviation();
return log(arg)/2;
}

} 
} 

#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif 


