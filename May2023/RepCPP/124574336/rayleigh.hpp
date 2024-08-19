
#ifndef BOOST_STATS_rayleigh_HPP
#define BOOST_STATS_rayleigh_HPP

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/config/no_tr1/cmath.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4702) 
#endif

#include <utility>

namespace boost{ namespace math{

namespace detail
{ 
template <class RealType, class Policy>
inline bool verify_sigma(const char* function, RealType sigma, RealType* presult, const Policy& pol)
{
if((sigma <= 0) || (!(boost::math::isfinite)(sigma)))
{
*presult = policies::raise_domain_error<RealType>(
function,
"The scale parameter \"sigma\" must be > 0 and finite, but was: %1%.", sigma, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool verify_rayleigh_x(const char* function, RealType x, RealType* presult, const Policy& pol)
{
if((x < 0) || (boost::math::isnan)(x))
{
*presult = policies::raise_domain_error<RealType>(
function,
"The random variable must be >= 0, but was: %1%.", x, pol);
return false;
}
return true;
} 
} 

template <class RealType = double, class Policy = policies::policy<> >
class rayleigh_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

rayleigh_distribution(RealType l_sigma = 1)
: m_sigma(l_sigma)
{
RealType err;
detail::verify_sigma("boost::math::rayleigh_distribution<%1%>::rayleigh_distribution", l_sigma, &err, Policy());
} 

RealType sigma()const
{ 
return m_sigma;
}

private:
RealType m_sigma;
}; 

typedef rayleigh_distribution<double> rayleigh;

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const rayleigh_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0), std::numeric_limits<RealType>::has_infinity ? std::numeric_limits<RealType>::infinity() : max_value<RealType>());
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const rayleigh_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0),  max_value<RealType>());
}

template <class RealType, class Policy>
inline RealType pdf(const rayleigh_distribution<RealType, Policy>& dist, const RealType& x)
{
BOOST_MATH_STD_USING 

RealType sigma = dist.sigma();
RealType result = 0;
static const char* function = "boost::math::pdf(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
{
return result;
}
if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
{
return result;
}
if((boost::math::isinf)(x))
{
return 0;
}
RealType sigmasqr = sigma * sigma;
result = x * (exp(-(x * x) / ( 2 * sigmasqr))) / sigmasqr;
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const rayleigh_distribution<RealType, Policy>& dist, const RealType& x)
{
BOOST_MATH_STD_USING 

RealType result = 0;
RealType sigma = dist.sigma();
static const char* function = "boost::math::cdf(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
{
return result;
}
if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
{
return result;
}
result = -boost::math::expm1(-x * x / ( 2 * sigma * sigma), Policy());
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const rayleigh_distribution<RealType, Policy>& dist, const RealType& p)
{
BOOST_MATH_STD_USING 

RealType result = 0;
RealType sigma = dist.sigma();
static const char* function = "boost::math::quantile(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
return result;
if(false == detail::check_probability(function, p, &result, Policy()))
return result;

if(p == 0)
{
return 0;
}
if(p == 1)
{
return policies::raise_overflow_error<RealType>(function, 0, Policy());
}
result = sqrt(-2 * sigma * sigma * boost::math::log1p(-p, Policy()));
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<rayleigh_distribution<RealType, Policy>, RealType>& c)
{
BOOST_MATH_STD_USING 

RealType result = 0;
RealType sigma = c.dist.sigma();
static const char* function = "boost::math::cdf(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
{
return result;
}
RealType x = c.param;
if(false == detail::verify_rayleigh_x(function, x, &result, Policy()))
{
return result;
}
RealType ea = x * x / (2 * sigma * sigma);
if (ea >= tools::max_value<RealType>())
return 0;
result =  exp(-ea);
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<rayleigh_distribution<RealType, Policy>, RealType>& c)
{
BOOST_MATH_STD_USING 

RealType result = 0;
RealType sigma = c.dist.sigma();
static const char* function = "boost::math::quantile(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
{
return result;
}
RealType q = c.param;
if(false == detail::check_probability(function, q, &result, Policy()))
{
return result;
}
if(q == 1)
{
return 0;
}
if(q == 0)
{
return policies::raise_overflow_error<RealType>(function, 0, Policy());
}
result = sqrt(-2 * sigma * sigma * log(q));
return result;
} 

template <class RealType, class Policy>
inline RealType mean(const rayleigh_distribution<RealType, Policy>& dist)
{
RealType result = 0;
RealType sigma = dist.sigma();
static const char* function = "boost::math::mean(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
{
return result;
}
using boost::math::constants::root_half_pi;
return sigma * root_half_pi<RealType>();
} 

template <class RealType, class Policy>
inline RealType variance(const rayleigh_distribution<RealType, Policy>& dist)
{
RealType result = 0;
RealType sigma = dist.sigma();
static const char* function = "boost::math::variance(const rayleigh_distribution<%1%>&, %1%)";
if(false == detail::verify_sigma(function, sigma, &result, Policy()))
{
return result;
}
using boost::math::constants::four_minus_pi;
return four_minus_pi<RealType>() * sigma * sigma / 2;
} 

template <class RealType, class Policy>
inline RealType mode(const rayleigh_distribution<RealType, Policy>& dist)
{
return dist.sigma();
}

template <class RealType, class Policy>
inline RealType median(const rayleigh_distribution<RealType, Policy>& dist)
{
using boost::math::constants::root_ln_four;
return root_ln_four<RealType>() * dist.sigma();
}

template <class RealType, class Policy>
inline RealType skewness(const rayleigh_distribution<RealType, Policy>& )
{
return static_cast<RealType>(0.63111065781893713819189935154422777984404221106391L);
}

template <class RealType, class Policy>
inline RealType kurtosis(const rayleigh_distribution<RealType, Policy>& )
{
return static_cast<RealType>(3.2450893006876380628486604106197544154170667057995L);
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const rayleigh_distribution<RealType, Policy>& )
{
return static_cast<RealType>(0.2450893006876380628486604106197544154170667057995L);
} 

template <class RealType, class Policy>
inline RealType entropy(const rayleigh_distribution<RealType, Policy>& dist)
{
using std::log;
return 1 + log(dist.sigma()*constants::one_div_root_two<RealType>()) + constants::euler<RealType>()/2;
}

} 
} 

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif 
