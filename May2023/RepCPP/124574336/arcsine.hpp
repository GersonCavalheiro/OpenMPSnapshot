







#ifndef BOOST_MATH_DIST_ARCSINE_HPP
#define BOOST_MATH_DIST_ARCSINE_HPP

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/distributions/complement.hpp> 
#include <boost/math/distributions/detail/common_error_handling.hpp> 
#include <boost/math/constants/constants.hpp>

#include <boost/math/special_functions/fpclassify.hpp> 

#if defined (BOOST_MSVC)
#  pragma warning(push)
#  pragma warning(disable: 4702) 
#endif

#include <utility>
#include <exception>  

namespace boost
{
namespace math
{
namespace arcsine_detail
{
template <class RealType, class Policy>
inline bool check_x_min(const char* function, const RealType& x, RealType* result, const Policy& pol)
{
if (!(boost::math::isfinite)(x))
{
*result = policies::raise_domain_error<RealType>(
function,
"x_min argument is %1%, but must be finite !", x, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_x_max(const char* function, const RealType& x, RealType* result, const Policy& pol)
{
if (!(boost::math::isfinite)(x))
{
*result = policies::raise_domain_error<RealType>(
function,
"x_max argument is %1%, but must be finite !", x, pol);
return false;
}
return true;
} 


template <class RealType, class Policy>
inline bool check_x_minmax(const char* function, const RealType& x_min, const RealType& x_max, RealType* result, const Policy& pol)
{ 
if (x_min >= x_max)
{
std::string msg = "x_max argument is %1%, but must be > x_min = " + lexical_cast<std::string>(x_min) + "!";
*result = policies::raise_domain_error<RealType>(
function,
msg.c_str(), x_max, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_prob(const char* function, const RealType& p, RealType* result, const Policy& pol)
{
if ((p < 0) || (p > 1) || !(boost::math::isfinite)(p))
{
*result = policies::raise_domain_error<RealType>(
function,
"Probability argument is %1%, but must be >= 0 and <= 1 !", p, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_x(const char* function, const RealType& x_min, const RealType& x_max, const RealType& x, RealType* result, const Policy& pol)
{ 
if (!(boost::math::isfinite)(x))
{
*result = policies::raise_domain_error<RealType>(
function,
"x argument is %1%, but must be finite !", x, pol);
return false;
}
if ((x < x_min) || (x > x_max))
{
*result = policies::raise_domain_error<RealType>(
function,
"x argument is %1%, but must be x_min < x < x_max !", x, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_dist(const char* function, const RealType& x_min, const RealType& x_max, RealType* result, const Policy& pol)
{ 
return check_x_min(function, x_min, result, pol)
&& check_x_max(function, x_max, result, pol)
&& check_x_minmax(function, x_min, x_max, result, pol);
} 

template <class RealType, class Policy>
inline bool check_dist_and_x(const char* function, const RealType& x_min, const RealType& x_max, RealType x, RealType* result, const Policy& pol)
{
return check_dist(function, x_min, x_max, result, pol)
&& arcsine_detail::check_x(function, x_min, x_max, x, result, pol);
} 

template <class RealType, class Policy>
inline bool check_dist_and_prob(const char* function, const RealType& x_min, const RealType& x_max, RealType p, RealType* result, const Policy& pol)
{
return check_dist(function, x_min, x_max, result, pol)
&& check_prob(function, p, result, pol);
} 

} 

template <class RealType = double, class Policy = policies::policy<> >
class arcsine_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

arcsine_distribution(RealType x_min = 0, RealType x_max = 1) : m_x_min(x_min), m_x_max(x_max)
{ 
RealType result;
arcsine_detail::check_dist(
"boost::math::arcsine_distribution<%1%>::arcsine_distribution",
m_x_min,
m_x_max,
&result, Policy());
} 
RealType x_min() const
{
return m_x_min;
}
RealType x_max() const
{
return m_x_max;
}

private:
RealType m_x_min; 
RealType m_x_max;
}; 

typedef arcsine_distribution<double> arcsine;


template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const arcsine_distribution<RealType, Policy>&  dist)
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(dist.x_min()), static_cast<RealType>(dist.x_max()));
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const arcsine_distribution<RealType, Policy>&  dist)
{ 
return std::pair<RealType, RealType>(static_cast<RealType>(dist.x_min()), static_cast<RealType>(dist.x_max()));
}

template <class RealType, class Policy>
inline RealType mean(const arcsine_distribution<RealType, Policy>& dist)
{ 
RealType result;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();

if (false == arcsine_detail::check_dist(
"boost::math::mean(arcsine_distribution<%1%> const&, %1% )",
x_min,
x_max,
&result, Policy())
)
{
return result;
}
return  (x_min + x_max) / 2;
} 

template <class RealType, class Policy>
inline RealType variance(const arcsine_distribution<RealType, Policy>& dist)
{ 
RealType result;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();
if (false == arcsine_detail::check_dist(
"boost::math::variance(arcsine_distribution<%1%> const&, %1% )",
x_min,
x_max,
&result, Policy())
)
{
return result;
}
return  (x_max - x_min) * (x_max - x_min) / 8;
} 

template <class RealType, class Policy>
inline RealType mode(const arcsine_distribution<RealType, Policy>& )
{ 
return policies::raise_domain_error<RealType>(
"boost::math::mode(arcsine_distribution<%1%>&)",
"The arcsine distribution has two modes at x_min and x_max: "
"so the return value is %1%.",
std::numeric_limits<RealType>::quiet_NaN(), Policy());
} 

template <class RealType, class Policy>
inline RealType median(const arcsine_distribution<RealType, Policy>& dist)
{ 
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();
RealType result;
if (false == arcsine_detail::check_dist(
"boost::math::median(arcsine_distribution<%1%> const&, %1% )",
x_min,
x_max,
&result, Policy())
)
{
return result;
}
return  (x_min + x_max) / 2;
}

template <class RealType, class Policy>
inline RealType skewness(const arcsine_distribution<RealType, Policy>& dist)
{
RealType result;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();

if (false == arcsine_detail::check_dist(
"boost::math::skewness(arcsine_distribution<%1%> const&, %1% )",
x_min,
x_max,
&result, Policy())
)
{
return result;
}
return 0;
} 

template <class RealType, class Policy>
inline RealType kurtosis_excess(const arcsine_distribution<RealType, Policy>& dist)
{
RealType result;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();

if (false == arcsine_detail::check_dist(
"boost::math::kurtosis_excess(arcsine_distribution<%1%> const&, %1% )",
x_min,
x_max,
&result, Policy())
)
{
return result;
}
result = -3;
return  result / 2;
} 

template <class RealType, class Policy>
inline RealType kurtosis(const arcsine_distribution<RealType, Policy>& dist)
{
RealType result;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();

if (false == arcsine_detail::check_dist(
"boost::math::kurtosis(arcsine_distribution<%1%> const&, %1% )",
x_min,
x_max,
&result, Policy())
)
{
return result;
}

return 3 + kurtosis_excess(dist);
} 

template <class RealType, class Policy>
inline RealType pdf(const arcsine_distribution<RealType, Policy>& dist, const RealType& xx)
{ 
BOOST_FPU_EXCEPTION_GUARD
BOOST_MATH_STD_USING 

static const char* function = "boost::math::pdf(arcsine_distribution<%1%> const&, %1%)";

RealType lo = dist.x_min();
RealType hi = dist.x_max();
RealType x = xx;

RealType result = 0; 
if (false == arcsine_detail::check_dist_and_x(
function,
lo, hi, x,
&result, Policy()))
{
return result;
}
using boost::math::constants::pi;
result = static_cast<RealType>(1) / (pi<RealType>() * sqrt((x - lo) * (hi - x)));
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const arcsine_distribution<RealType, Policy>& dist, const RealType& x)
{ 
BOOST_MATH_STD_USING 

static const char* function = "boost::math::cdf(arcsine_distribution<%1%> const&, %1%)";

RealType x_min = dist.x_min();
RealType x_max = dist.x_max();

RealType result = 0;
if (false == arcsine_detail::check_dist_and_x(
function,
x_min, x_max, x,
&result, Policy()))
{
return result;
}
if (x == x_min)
{
return 0;
}
else if (x == x_max)
{
return 1;
}
using boost::math::constants::pi;
result = static_cast<RealType>(2) * asin(sqrt((x - x_min) / (x_max - x_min))) / pi<RealType>();
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<arcsine_distribution<RealType, Policy>, RealType>& c)
{ 
BOOST_MATH_STD_USING 
static const char* function = "boost::math::cdf(arcsine_distribution<%1%> const&, %1%)";

RealType x = c.param;
arcsine_distribution<RealType, Policy> const& dist = c.dist;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();

RealType result = 0;
if (false == arcsine_detail::check_dist_and_x(
function,
x_min, x_max, x,
&result, Policy()))
{
return result;
}
if (x == x_min)
{
return 0;
}
else if (x == x_max)
{
return 1;
}
using boost::math::constants::pi;
result = static_cast<RealType>(2) * acos(sqrt((x - x_min) / (x_max - x_min))) / pi<RealType>();
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const arcsine_distribution<RealType, Policy>& dist, const RealType& p)
{ 
BOOST_MATH_STD_USING 

using boost::math::constants::half_pi;

static const char* function = "boost::math::quantile(arcsine_distribution<%1%> const&, %1%)";

RealType result = 0; 
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();
if (false == arcsine_detail::check_dist_and_prob(
function,
x_min, x_max, p,
&result, Policy()))
{
return result;
}
if (p == 0)
{
return 0;
}
if (p == 1)
{
return 1;
}

RealType sin2hpip = sin(half_pi<RealType>() * p);
RealType sin2hpip2 = sin2hpip * sin2hpip;
result = -x_min * sin2hpip2 + x_min + x_max * sin2hpip2;

return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<arcsine_distribution<RealType, Policy>, RealType>& c)
{ 
BOOST_MATH_STD_USING 

using boost::math::constants::half_pi;
static const char* function = "boost::math::quantile(arcsine_distribution<%1%> const&, %1%)";

RealType q = c.param;
const arcsine_distribution<RealType, Policy>& dist = c.dist;
RealType result = 0;
RealType x_min = dist.x_min();
RealType x_max = dist.x_max();
if (false == arcsine_detail::check_dist_and_prob(
function,
x_min,
x_max,
q,
&result, Policy()))
{
return result;
}
if (q == 1)
{
return 0;
}
if (q == 0)
{
return 1;
}
RealType cos2hpip = cos(half_pi<RealType>() * q);
RealType cos2hpip2 = cos2hpip * cos2hpip;
result = -x_min * cos2hpip2 + x_min + x_max * cos2hpip2;

return result;
} 

} 
} 

#include <boost/math/distributions/detail/derived_accessors.hpp>

#if defined (BOOST_MSVC)
# pragma warning(pop)
#endif

#endif 
