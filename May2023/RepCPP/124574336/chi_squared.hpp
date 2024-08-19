

#ifndef BOOST_MATH_DISTRIBUTIONS_CHI_SQUARED_HPP
#define BOOST_MATH_DISTRIBUTIONS_CHI_SQUARED_HPP

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/special_functions/gamma.hpp> 
#include <boost/math/distributions/complement.hpp> 
#include <boost/math/distributions/detail/common_error_handling.hpp> 
#include <boost/math/special_functions/fpclassify.hpp>

#include <utility>

namespace boost{ namespace math{

template <class RealType = double, class Policy = policies::policy<> >
class chi_squared_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

chi_squared_distribution(RealType i) : m_df(i)
{
RealType result;
detail::check_df(
"boost::math::chi_squared_distribution<%1%>::chi_squared_distribution", m_df, &result, Policy());
} 

RealType degrees_of_freedom()const
{
return m_df;
}

static RealType find_degrees_of_freedom(
RealType difference_from_variance,
RealType alpha,
RealType beta,
RealType variance,
RealType hint = 100);

private:
RealType m_df; 
}; 

typedef chi_squared_distribution<double> chi_squared;

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)
#endif

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const chi_squared_distribution<RealType, Policy>& )
{ 
if (std::numeric_limits<RealType>::has_infinity)
{ 
return std::pair<RealType, RealType>(static_cast<RealType>(0), std::numeric_limits<RealType>::infinity()); 
}
else
{
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(static_cast<RealType>(0), max_value<RealType>()); 
}
}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const chi_squared_distribution<RealType, Policy>& )
{ 
return std::pair<RealType, RealType>(static_cast<RealType>(0), tools::max_value<RealType>()); 
}

template <class RealType, class Policy>
RealType pdf(const chi_squared_distribution<RealType, Policy>& dist, const RealType& chi_square)
{
BOOST_MATH_STD_USING  
RealType degrees_of_freedom = dist.degrees_of_freedom();
RealType error_result;

static const char* function = "boost::math::pdf(const chi_squared_distribution<%1%>&, %1%)";

if(false == detail::check_df(
function, degrees_of_freedom, &error_result, Policy()))
return error_result;

if((chi_square < 0) || !(boost::math::isfinite)(chi_square))
{
return policies::raise_domain_error<RealType>(
function, "Chi Square parameter was %1%, but must be > 0 !", chi_square, Policy());
}

if(chi_square == 0)
{
if(degrees_of_freedom < 2)
{
return policies::raise_overflow_error<RealType>(
function, 0, Policy());
}
else if(degrees_of_freedom == 2)
{
return 0.5f;
}
else
{
return 0;
}
}

return gamma_p_derivative(degrees_of_freedom / 2, chi_square / 2, Policy()) / 2;
} 

template <class RealType, class Policy>
inline RealType cdf(const chi_squared_distribution<RealType, Policy>& dist, const RealType& chi_square)
{
RealType degrees_of_freedom = dist.degrees_of_freedom();
RealType error_result;
static const char* function = "boost::math::cdf(const chi_squared_distribution<%1%>&, %1%)";

if(false == detail::check_df(
function, degrees_of_freedom, &error_result, Policy()))
return error_result;

if((chi_square < 0) || !(boost::math::isfinite)(chi_square))
{
return policies::raise_domain_error<RealType>(
function, "Chi Square parameter was %1%, but must be > 0 !", chi_square, Policy());
}

return boost::math::gamma_p(degrees_of_freedom / 2, chi_square / 2, Policy());
} 

template <class RealType, class Policy>
inline RealType quantile(const chi_squared_distribution<RealType, Policy>& dist, const RealType& p)
{
RealType degrees_of_freedom = dist.degrees_of_freedom();
static const char* function = "boost::math::quantile(const chi_squared_distribution<%1%>&, %1%)";
RealType error_result;
if(false ==
(
detail::check_df(function, degrees_of_freedom, &error_result, Policy())
&& detail::check_probability(function, p, &error_result, Policy()))
)
return error_result;

return 2 * boost::math::gamma_p_inv(degrees_of_freedom / 2, p, Policy());
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<chi_squared_distribution<RealType, Policy>, RealType>& c)
{
RealType const& degrees_of_freedom = c.dist.degrees_of_freedom();
RealType const& chi_square = c.param;
static const char* function = "boost::math::cdf(const chi_squared_distribution<%1%>&, %1%)";
RealType error_result;
if(false == detail::check_df(
function, degrees_of_freedom, &error_result, Policy()))
return error_result;

if((chi_square < 0) || !(boost::math::isfinite)(chi_square))
{
return policies::raise_domain_error<RealType>(
function, "Chi Square parameter was %1%, but must be > 0 !", chi_square, Policy());
}

return boost::math::gamma_q(degrees_of_freedom / 2, chi_square / 2, Policy());
}

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<chi_squared_distribution<RealType, Policy>, RealType>& c)
{
RealType const& degrees_of_freedom = c.dist.degrees_of_freedom();
RealType const& q = c.param;
static const char* function = "boost::math::quantile(const chi_squared_distribution<%1%>&, %1%)";
RealType error_result;
if(false == (
detail::check_df(function, degrees_of_freedom, &error_result, Policy())
&& detail::check_probability(function, q, &error_result, Policy()))
)
return error_result;

return 2 * boost::math::gamma_q_inv(degrees_of_freedom / 2, q, Policy());
}

template <class RealType, class Policy>
inline RealType mean(const chi_squared_distribution<RealType, Policy>& dist)
{ 
return dist.degrees_of_freedom();
} 

template <class RealType, class Policy>
inline RealType variance(const chi_squared_distribution<RealType, Policy>& dist)
{ 
return 2 * dist.degrees_of_freedom();
} 

template <class RealType, class Policy>
inline RealType mode(const chi_squared_distribution<RealType, Policy>& dist)
{
RealType df = dist.degrees_of_freedom();
static const char* function = "boost::math::mode(const chi_squared_distribution<%1%>&)";

if(df < 2)
return policies::raise_domain_error<RealType>(
function,
"Chi-Squared distribution only has a mode for degrees of freedom >= 2, but got degrees of freedom = %1%.",
df, Policy());
return df - 2;
}


template <class RealType, class Policy>
inline RealType skewness(const chi_squared_distribution<RealType, Policy>& dist)
{
BOOST_MATH_STD_USING 
RealType df = dist.degrees_of_freedom();
return sqrt (8 / df);  
}

template <class RealType, class Policy>
inline RealType kurtosis(const chi_squared_distribution<RealType, Policy>& dist)
{
RealType df = dist.degrees_of_freedom();
return 3 + 12 / df;
}

template <class RealType, class Policy>
inline RealType kurtosis_excess(const chi_squared_distribution<RealType, Policy>& dist)
{
RealType df = dist.degrees_of_freedom();
return 12 / df;
}

namespace detail
{

template <class RealType, class Policy>
struct df_estimator
{
df_estimator(RealType a, RealType b, RealType variance, RealType delta)
: alpha(a), beta(b), ratio(delta/variance)
{ 
}

RealType operator()(const RealType& df)
{
if(df <= tools::min_value<RealType>())
return 1;
chi_squared_distribution<RealType, Policy> cs(df);

RealType result;
if(ratio > 0)
{
RealType r = 1 + ratio;
result = cdf(cs, quantile(complement(cs, alpha)) / r) - beta;
}
else
{ 
RealType r = 1 + ratio;
result = cdf(complement(cs, quantile(cs, alpha) / r)) - beta;
}
return result;
}
private:
RealType alpha;
RealType beta;
RealType ratio; 
};

} 

template <class RealType, class Policy>
RealType chi_squared_distribution<RealType, Policy>::find_degrees_of_freedom(
RealType difference_from_variance,
RealType alpha,
RealType beta,
RealType variance,
RealType hint)
{
static const char* function = "boost::math::chi_squared_distribution<%1%>::find_degrees_of_freedom(%1%,%1%,%1%,%1%,%1%)";
RealType error_result;
if(false ==
detail::check_probability(function, alpha, &error_result, Policy())
&& detail::check_probability(function, beta, &error_result, Policy()))
{ 
return error_result;
}

if(hint <= 0)
{ 
hint = 1;
}

detail::df_estimator<RealType, Policy> f(alpha, beta, variance, difference_from_variance);
tools::eps_tolerance<RealType> tol(policies::digits<RealType, Policy>());
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
std::pair<RealType, RealType> r =
tools::bracket_and_solve_root(f, hint, RealType(2), false, tol, max_iter, Policy());
RealType result = r.first + (r.second - r.first) / 2;
if(max_iter >= policies::get_max_root_iterations<Policy>())
{
policies::raise_evaluation_error<RealType>(function, "Unable to locate solution in a reasonable time:"
" either there is no answer to how many degrees of freedom are required"
" or the answer is infinite.  Current best guess is %1%", result, Policy());
}
return result;
}

} 
} 

#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif 
