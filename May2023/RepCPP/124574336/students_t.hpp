

#ifndef BOOST_STATS_STUDENTS_T_HPP
#define BOOST_STATS_STUDENTS_T_HPP


#include <boost/math/distributions/fwd.hpp>
#include <boost/math/special_functions/beta.hpp> 
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/distributions/complement.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <boost/math/distributions/normal.hpp> 

#include <utility>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4702) 
#endif

namespace boost { namespace math {

template <class RealType = double, class Policy = policies::policy<> >
class students_t_distribution
{
public:
typedef RealType value_type;
typedef Policy policy_type;

students_t_distribution(RealType df) : df_(df)
{ 
RealType result;
detail::check_df_gt0_to_inf( 
"boost::math::students_t_distribution<%1%>::students_t_distribution", df_, &result, Policy());
} 

RealType degrees_of_freedom()const
{
return df_;
}

static RealType find_degrees_of_freedom(
RealType difference_from_mean,
RealType alpha,
RealType beta,
RealType sd,
RealType hint = 100);

private:
RealType df_;  
};

typedef students_t_distribution<double> students_t; 

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> range(const students_t_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(((::std::numeric_limits<RealType>::is_specialized & ::std::numeric_limits<RealType>::has_infinity) ? -std::numeric_limits<RealType>::infinity() : -max_value<RealType>()), ((::std::numeric_limits<RealType>::is_specialized & ::std::numeric_limits<RealType>::has_infinity) ? +std::numeric_limits<RealType>::infinity() : +max_value<RealType>()));
}

template <class RealType, class Policy>
inline const std::pair<RealType, RealType> support(const students_t_distribution<RealType, Policy>& )
{ 
using boost::math::tools::max_value;
return std::pair<RealType, RealType>(((::std::numeric_limits<RealType>::is_specialized & ::std::numeric_limits<RealType>::has_infinity) ? -std::numeric_limits<RealType>::infinity() : -max_value<RealType>()), ((::std::numeric_limits<RealType>::is_specialized & ::std::numeric_limits<RealType>::has_infinity) ? +std::numeric_limits<RealType>::infinity() : +max_value<RealType>()));
}

template <class RealType, class Policy>
inline RealType pdf(const students_t_distribution<RealType, Policy>& dist, const RealType& x)
{
BOOST_FPU_EXCEPTION_GUARD
BOOST_MATH_STD_USING  

RealType error_result;
if(false == detail::check_x_not_NaN(
"boost::math::pdf(const students_t_distribution<%1%>&, %1%)", x, &error_result, Policy()))
return error_result;
RealType df = dist.degrees_of_freedom();
if(false == detail::check_df_gt0_to_inf( 
"boost::math::pdf(const students_t_distribution<%1%>&, %1%)", df, &error_result, Policy()))
return error_result;

RealType result;
if ((boost::math::isinf)(x))
{ 
result = static_cast<RealType>(0);
return result;
}
RealType limit = policies::get_epsilon<RealType, Policy>();
limit = static_cast<RealType>(1) / limit; 
if (df > limit)
{ 
normal_distribution<RealType, Policy> n(0, 1); 
result = pdf(n, x);
}
else
{ 
RealType basem1 = x * x / df;
if(basem1 < 0.125)
{
result = exp(-boost::math::log1p(basem1, Policy()) * (1+df) / 2);
}
else
{
result = pow(1 / (1 + basem1), (df + 1) / 2);
}
result /= sqrt(df) * boost::math::beta(df / 2, RealType(0.5f), Policy());
}
return result;
} 

template <class RealType, class Policy>
inline RealType cdf(const students_t_distribution<RealType, Policy>& dist, const RealType& x)
{
RealType error_result;
RealType df = dist.degrees_of_freedom();
if (false == detail::check_df_gt0_to_inf(  
"boost::math::cdf(const students_t_distribution<%1%>&, %1%)", df, &error_result, Policy()))
{
return error_result;
}
if(false == detail::check_x_not_NaN(
"boost::math::cdf(const students_t_distribution<%1%>&, %1%)", x, &error_result, Policy()))
{ 
return error_result;
}
if (x == 0)
{ 
return static_cast<RealType>(0.5);
}
if ((boost::math::isinf)(x))
{ 
return ((x < 0) ? static_cast<RealType>(0) : static_cast<RealType>(1));
}

RealType limit = policies::get_epsilon<RealType, Policy>();
limit = static_cast<RealType>(1) / limit; 
if (df > limit)
{ 
normal_distribution<RealType, Policy> n(0, 1); 
RealType result = cdf(n, x);
return result;
}
else
{ 
RealType x2 = x * x;
RealType probability;
if(df > 2 * x2)
{
RealType z = x2 / (df + x2);
probability = ibetac(static_cast<RealType>(0.5), df / 2, z, Policy()) / 2;
}
else
{
RealType z = df / (df + x2);
probability = ibeta(df / 2, static_cast<RealType>(0.5), z, Policy()) / 2;
}
return (x > 0 ? 1   - probability : probability);
}
} 

template <class RealType, class Policy>
inline RealType quantile(const students_t_distribution<RealType, Policy>& dist, const RealType& p)
{
BOOST_MATH_STD_USING 
RealType probability = p;

RealType df = dist.degrees_of_freedom();
static const char* function = "boost::math::quantile(const students_t_distribution<%1%>&, %1%)";
RealType error_result;
if(false == (detail::check_df_gt0_to_inf( 
function, df, &error_result, Policy())
&& detail::check_probability(function, probability, &error_result, Policy())))
return error_result;
if (probability == 0)
return -policies::raise_overflow_error<RealType>(function, 0, Policy());
if (probability == 1)
return policies::raise_overflow_error<RealType>(function, 0, Policy());
if (probability == static_cast<RealType>(0.5))
return 0;  
#if 0
probability = (probability > 0.5) ? 1 - probability : probability;
RealType t, x, y;
x = ibeta_inv(degrees_of_freedom / 2, RealType(0.5), 2 * probability, &y);
if(degrees_of_freedom * y > tools::max_value<RealType>() * x)
t = tools::overflow_error<RealType>(function);
else
t = sqrt(degrees_of_freedom * y / x);
if(p < 0.5)
t = -t;

return t;
#endif
return boost::math::detail::fast_students_t_quantile(df, probability, Policy());
} 

template <class RealType, class Policy>
inline RealType cdf(const complemented2_type<students_t_distribution<RealType, Policy>, RealType>& c)
{
return cdf(c.dist, -c.param);
}

template <class RealType, class Policy>
inline RealType quantile(const complemented2_type<students_t_distribution<RealType, Policy>, RealType>& c)
{
return -quantile(c.dist, c.param);
}

namespace detail{
template <class RealType, class Policy>
struct sample_size_func
{
sample_size_func(RealType a, RealType b, RealType s, RealType d)
: alpha(a), beta(b), ratio(s*s/(d*d)) {}

RealType operator()(const RealType& df)
{
if(df <= tools::min_value<RealType>())
{ 
return 1;
}
students_t_distribution<RealType, Policy> t(df);
RealType qa = quantile(complement(t, alpha));
RealType qb = quantile(complement(t, beta));
qa += qb;
qa *= qa;
qa *= ratio;
qa -= (df + 1);
return qa;
}
RealType alpha, beta, ratio;
};

}  

template <class RealType, class Policy>
RealType students_t_distribution<RealType, Policy>::find_degrees_of_freedom(
RealType difference_from_mean,
RealType alpha,
RealType beta,
RealType sd,
RealType hint)
{
static const char* function = "boost::math::students_t_distribution<%1%>::find_degrees_of_freedom";
RealType error_result;
if(false == detail::check_probability(
function, alpha, &error_result, Policy())
&& detail::check_probability(function, beta, &error_result, Policy()))
return error_result;

if(hint <= 0)
hint = 1;

detail::sample_size_func<RealType, Policy> f(alpha, beta, sd, difference_from_mean);
tools::eps_tolerance<RealType> tol(policies::digits<RealType, Policy>());
boost::uintmax_t max_iter = policies::get_max_root_iterations<Policy>();
std::pair<RealType, RealType> r = tools::bracket_and_solve_root(f, hint, RealType(2), false, tol, max_iter, Policy());
RealType result = r.first + (r.second - r.first) / 2;
if(max_iter >= policies::get_max_root_iterations<Policy>())
{
return policies::raise_evaluation_error<RealType>(function, "Unable to locate solution in a reasonable time:"
" either there is no answer to how many degrees of freedom are required"
" or the answer is infinite.  Current best guess is %1%", result, Policy());
}
return result;
}

template <class RealType, class Policy>
inline RealType mode(const students_t_distribution<RealType, Policy>& )
{
return 0; 
}

template <class RealType, class Policy>
inline RealType median(const students_t_distribution<RealType, Policy>& )
{
return 0; 
}


template <class RealType, class Policy>
inline RealType mean(const students_t_distribution<RealType, Policy>& dist)
{  
RealType df = dist.degrees_of_freedom();
if(((boost::math::isnan)(df)) || (df <= 1) ) 
{ 
return policies::raise_domain_error<RealType>(
"boost::math::mean(students_t_distribution<%1%> const&, %1%)",
"Mean is undefined for degrees of freedom < 1 but got %1%.", df, Policy());
return std::numeric_limits<RealType>::quiet_NaN();
}
return 0;
} 

template <class RealType, class Policy>
inline RealType variance(const students_t_distribution<RealType, Policy>& dist)
{ 
RealType df = dist.degrees_of_freedom();
if ((boost::math::isnan)(df) || (df <= 2))
{ 
return policies::raise_domain_error<RealType>(
"boost::math::variance(students_t_distribution<%1%> const&, %1%)",
"variance is undefined for degrees of freedom <= 2, but got %1%.",
df, Policy());
return std::numeric_limits<RealType>::quiet_NaN(); 
}
if ((boost::math::isinf)(df))
{ 
return 1;
}
RealType limit = policies::get_epsilon<RealType, Policy>();
limit = static_cast<RealType>(1) / limit; 
if (df > limit)
{ 
return 1;
}
else
{
return df / (df - 2);
}
} 

template <class RealType, class Policy>
inline RealType skewness(const students_t_distribution<RealType, Policy>& dist)
{
RealType df = dist.degrees_of_freedom();
if( ((boost::math::isnan)(df)) || (dist.degrees_of_freedom() <= 3))
{ 
return policies::raise_domain_error<RealType>(
"boost::math::skewness(students_t_distribution<%1%> const&, %1%)",
"Skewness is undefined for degrees of freedom <= 3, but got %1%.",
dist.degrees_of_freedom(), Policy());
return std::numeric_limits<RealType>::quiet_NaN();
}
return 0; 
} 

template <class RealType, class Policy>
inline RealType kurtosis(const students_t_distribution<RealType, Policy>& dist)
{
RealType df = dist.degrees_of_freedom();
if(((boost::math::isnan)(df)) || (df <= 4))
{ 
return policies::raise_domain_error<RealType>(
"boost::math::kurtosis(students_t_distribution<%1%> const&, %1%)",
"Kurtosis is undefined for degrees of freedom <= 4, but got %1%.",
df, Policy());
return std::numeric_limits<RealType>::quiet_NaN(); 
}
if ((boost::math::isinf)(df))
{ 
return 3;
}
RealType limit = policies::get_epsilon<RealType, Policy>();
limit = static_cast<RealType>(1) / limit; 
if (df > limit)
{ 
return 3;
}
else
{
return 6 / (df - 4) + 3;
}
} 

template <class RealType, class Policy>
inline RealType kurtosis_excess(const students_t_distribution<RealType, Policy>& dist)
{

RealType df = dist.degrees_of_freedom();
if(((boost::math::isnan)(df)) || (df <= 4))
{ 
return policies::raise_domain_error<RealType>(
"boost::math::kurtosis_excess(students_t_distribution<%1%> const&, %1%)",
"Kurtosis_excess is undefined for degrees of freedom <= 4, but got %1%.",
df, Policy());
return std::numeric_limits<RealType>::quiet_NaN(); 
}
if ((boost::math::isinf)(df))
{ 
return 0;
}
RealType limit = policies::get_epsilon<RealType, Policy>();
limit = static_cast<RealType>(1) / limit; 
if (df > limit)
{ 
return 0;
}
else
{
return 6 / (df - 4);
}
}

template <class RealType, class Policy>
inline RealType entropy(const students_t_distribution<RealType, Policy>& dist)
{
using std::log;
using std::sqrt;
RealType v = dist.degrees_of_freedom();
RealType vp1 = (v+1)/2;
RealType vd2 = v/2;

return vp1*(digamma(vp1) - digamma(vd2)) + log(sqrt(v)*beta(vd2, RealType(1)/RealType(2)));
}

} 
} 

#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#include <boost/math/distributions/detail/derived_accessors.hpp>

#endif 
