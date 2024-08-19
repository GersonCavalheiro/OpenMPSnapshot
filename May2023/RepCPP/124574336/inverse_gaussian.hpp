

#ifndef BOOST_STATS_INVERSE_GAUSSIAN_HPP
#define BOOST_STATS_INVERSE_GAUSSIAN_HPP

#ifdef _MSC_VER
#pragma warning(disable: 4512) 
#endif











#include <boost/math/special_functions/erf.hpp> 
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/gamma.hpp> 

#include <boost/math/tools/tuple.hpp>
#include <boost/math/tools/roots.hpp>

#include <utility>

namespace boost{ namespace math{

template <class RealType = double, class Policy = policies::policy<> >
class inverse_gaussian_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

inverse_gaussian_distribution(RealType l_mean = 1, RealType l_scale = 1)
: m_mean(l_mean), m_scale(l_scale)
{ 
static const char* function = "boost::math::inverse_gaussian_distribution<%1%>::inverse_gaussian_distribution";

RealType result;
detail::check_scale(function, l_scale, &result, Policy());
detail::check_location(function, l_mean, &result, Policy());
detail::check_x_gt0(function, l_mean, &result, Policy());
}

RealType mean()const
{ 
return m_mean; 
}

RealType location()const
{ 
return m_mean;
}
RealType scale()const
{ 
return m_scale;
}

RealType shape()const
{ 
return m_scale / m_mean;
}

private:
RealType m_mean;  
RealType m_scale;    
}; 

typedef inverse_gaussian_distribution<double> inverse_gaussian;

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const inverse_gaussian_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0.), max_value<RealType>()); 
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const inverse_gaussian_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0.),  max_value<RealType>()); 
}

template <class RealType, class Policy>
inline RealType pdf(const inverse_gaussian_distribution<RealType, Policy>& dist, const RealType& x)
{ 
BOOST_MATH_STD_USING  

RealType scale = dist.scale();
RealType mean = dist.mean();
RealType result = 0;
static const char* function = "boost::math::pdf(const inverse_gaussian_distribution<%1%>&, %1%)";
if(false == detail::check_scale(function, scale, &result, Policy()))
{
return result;
}
if(false == detail::check_location(function, mean, &result, Policy()))
{
return result;
}
if(false == detail::check_x_gt0(function, mean, &result, Policy()))
{
return result;
}
if(false == detail::check_positive_x(function, x, &result, Policy()))
{
return result;
}

if (x == 0)
{
return 0; 
}

result =
sqrt(scale / (constants::two_pi<RealType>() * x * x * x))
* exp(-scale * (x - mean) * (x - mean) / (2 * x * mean * mean));
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const inverse_gaussian_distribution<RealType, Policy>& dist, const RealType& x)
{ 
BOOST_MATH_STD_USING  

RealType scale = dist.scale();
RealType mean = dist.mean();
static const char* function = "boost::math::cdf(const inverse_gaussian_distribution<%1%>&, %1%)";
RealType result = 0;
if(false == detail::check_scale(function, scale, &result, Policy()))
{
return result;
}
if(false == detail::check_location(function, mean, &result, Policy()))
{
return result;
}
if (false == detail::check_x_gt0(function, mean, &result, Policy()))
{
return result;
}
if(false == detail::check_positive_x(function, x, &result, Policy()))
{
return result;
}
if (x == 0)
{
return 0; 
}

normal_distribution<RealType> n01;

RealType n0 = sqrt(scale / x);
n0 *= ((x / mean) -1);
RealType n1 = cdf(n01, n0);
RealType expfactor = exp(2 * scale / mean);
RealType n3 = - sqrt(scale / x);
n3 *= (x / mean) + 1;
RealType n4 = cdf(n01, n3);
result = n1 + expfactor * n4;
return result;
} 

template <class RealType, class Policy>
struct inverse_gaussian_quantile_functor
{ 

inverse_gaussian_quantile_functor(const boost::math::inverse_gaussian_distribution<RealType, Policy> dist, RealType const& p)
: distribution(dist), prob(p)
{
}
boost::math::tuple<RealType, RealType> operator()(RealType const& x)
{
RealType c = cdf(distribution, x);
RealType fx = c - prob;  
RealType dx = pdf(distribution, x); 
return boost::math::make_tuple(fx, dx);
}
private:
const boost::math::inverse_gaussian_distribution<RealType, Policy> distribution;
RealType prob; 
};

template <class RealType, class Policy>
struct inverse_gaussian_quantile_complement_functor
{ 
inverse_gaussian_quantile_complement_functor(const boost::math::inverse_gaussian_distribution<RealType, Policy> dist, RealType const& p)
: distribution(dist), prob(p)
{
}
boost::math::tuple<RealType, RealType> operator()(RealType const& x)
{
RealType c = cdf(complement(distribution, x));
RealType fx = c - prob;  
RealType dx = -pdf(distribution, x); 
return boost::math::make_tuple(fx, dx);
}
private:
const boost::math::inverse_gaussian_distribution<RealType, Policy> distribution;
RealType prob; 
};

namespace detail
{
template <class RealType>
inline RealType guess_ig(RealType p, RealType mu = 1, RealType lambda = 1)
{ 
BOOST_MATH_STD_USING
using boost::math::policies::policy;
using boost::math::policies::overflow_error;
using boost::math::policies::ignore_error;

typedef policy<
overflow_error<ignore_error> 
> no_overthrow_policy;

RealType x; 
RealType phi = lambda / mu;
if (phi > 2.)
{ 

normal_distribution<RealType, no_overthrow_policy> n01;
x = mu * exp(quantile(n01, p) / sqrt(phi) - 1/(2 * phi));
}
else
{ 
using boost::math::gamma_distribution;

typedef gamma_distribution<RealType, no_overthrow_policy> gamma_nooverflow;

gamma_nooverflow g(static_cast<RealType>(0.5), static_cast<RealType>(1.));

RealType qg = quantile(complement(g, p));
x = lambda / (qg * 2);
if (x > mu/2) 
{ 
RealType q = quantile(g, p);
x = mu * exp(q / sqrt(phi) - 1/(2 * phi));
}
}
return x;
}  
} 

template <class RealType, class Policy>
inline RealType quantile(const inverse_gaussian_distribution<RealType, Policy>& dist, const RealType& p)
{
BOOST_MATH_STD_USING  

RealType mean = dist.mean();
RealType scale = dist.scale();
static const char* function = "boost::math::quantile(const inverse_gaussian_distribution<%1%>&, %1%)";

RealType result = 0;
if(false == detail::check_scale(function, scale, &result, Policy()))
return result;
if(false == detail::check_location(function, mean, &result, Policy()))
return result;
if (false == detail::check_x_gt0(function, mean, &result, Policy()))
return result;
if(false == detail::check_probability(function, p, &result, Policy()))
return result;
if (p == 0)
{
return 0; 
}
if (p == 1)
{ 
result = policies::raise_overflow_error<RealType>(function,
"probability parameter is 1, but must be < 1!", Policy());
return result; 
}

RealType guess = detail::guess_ig(p, dist.mean(), dist.scale());
using boost::math::tools::max_value;

RealType min = 0.; 
RealType max = max_value<RealType>();
int get_digits = policies::digits<RealType, Policy>();
boost::uintmax_t m = policies::get_max_root_iterations<Policy>(); 
using boost::math::tools::newton_raphson_iterate;
result =
newton_raphson_iterate(inverse_gaussian_quantile_functor<RealType, Policy>(dist, p), guess, min, max, get_digits, m);
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<inverse_gaussian_distribution<RealType, Policy>, RealType>& c)
{
BOOST_MATH_STD_USING  

RealType scale = c.dist.scale();
RealType mean = c.dist.mean();
RealType x = c.param;
static const char* function = "boost::math::cdf(const complement(inverse_gaussian_distribution<%1%>&), %1%)";
RealType result = 0;
if(false == detail::check_scale(function, scale, &result, Policy()))
return result;
if(false == detail::check_location(function, mean, &result, Policy()))
return result;
if (false == detail::check_x_gt0(function, mean, &result, Policy()))
return result;
if(false == detail::check_positive_x(function, x, &result, Policy()))
return result;

normal_distribution<RealType> n01;
RealType n0 = sqrt(scale / x);
n0 *= ((x / mean) -1);
RealType cdf_1 = cdf(complement(n01, n0));

RealType expfactor = exp(2 * scale / mean);
RealType n3 = - sqrt(scale / x);
n3 *= (x / mean) + 1;

RealType n6 = cdf(complement(n01, +sqrt(scale/x) * ((x /mean) + 1)));
result = cdf_1 - expfactor * n6; 
return result;
} 

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<inverse_gaussian_distribution<RealType, Policy>, RealType>& c)
{
BOOST_MATH_STD_USING  

RealType scale = c.dist.scale();
RealType mean = c.dist.mean();
static const char* function = "boost::math::quantile(const complement(inverse_gaussian_distribution<%1%>&), %1%)";
RealType result = 0;
if(false == detail::check_scale(function, scale, &result, Policy()))
return result;
if(false == detail::check_location(function, mean, &result, Policy()))
return result;
if (false == detail::check_x_gt0(function, mean, &result, Policy()))
return result;
RealType q = c.param;
if(false == detail::check_probability(function, q, &result, Policy()))
return result;

RealType guess = detail::guess_ig(q, mean, scale);
using boost::math::tools::max_value;

RealType min = 0.; 
RealType max = max_value<RealType>();
int get_digits = policies::digits<RealType, Policy>();
boost::uintmax_t m = policies::get_max_root_iterations<Policy>();
using boost::math::tools::newton_raphson_iterate;
result =
newton_raphson_iterate(inverse_gaussian_quantile_complement_functor<RealType, Policy>(c.dist, q), guess, min, max, get_digits, m);
return result;
} 

template <class RealType, class Policy>
inline RealType mean(const inverse_gaussian_distribution<RealType, Policy>& dist)
{ 
return dist.mean();
}

template <class RealType, class Policy>
inline RealType scale(const inverse_gaussian_distribution<RealType, Policy>& dist)
{ 
return dist.scale();
}

template <class RealType, class Policy>
inline RealType shape(const inverse_gaussian_distribution<RealType, Policy>& dist)
{ 
return dist.shape();
}

template <class RealType, class Policy>
inline RealType standard_deviation(const inverse_gaussian_distribution<RealType, Policy>& dist)
{
BOOST_MATH_STD_USING
RealType scale = dist.scale();
RealType mean = dist.mean();
RealType result = sqrt(mean * mean * mean / scale);
return result;
}

template <class RealType, class Policy>
inline RealType mode(const inverse_gaussian_distribution<RealType, Policy>& dist)
{
BOOST_MATH_STD_USING
RealType scale = dist.scale();
RealType  mean = dist.mean();
RealType result = mean * (sqrt(1 + (9 * mean * mean)/(4 * scale * scale)) 
- 3 * mean / (2 * scale));
return result;
}

template <class RealType, class Policy>
inline RealType skewness(const inverse_gaussian_distribution<RealType, Policy>& dist)
{
BOOST_MATH_STD_USING
RealType scale = dist.scale();
RealType  mean = dist.mean();
RealType result = 3 * sqrt(mean/scale);
return result;
}

template <class RealType, class Policy>
inline RealType kurtosis(const inverse_gaussian_distribution<RealType, Policy>& dist)
{
RealType scale = dist.scale();
RealType  mean = dist.mean();
RealType result = 15 * mean / scale -3;
return result;
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const inverse_gaussian_distribution<RealType, Policy>& dist)
{
RealType scale = dist.scale();
RealType  mean = dist.mean();
RealType result = 15 * mean / scale;
return result;
}

} 
} 

#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif 


