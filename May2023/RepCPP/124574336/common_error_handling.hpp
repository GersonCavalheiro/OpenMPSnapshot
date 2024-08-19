

#ifndef BOOST_MATH_DISTRIBUTIONS_COMMON_ERROR_HANDLING_HPP
#define BOOST_MATH_DISTRIBUTIONS_COMMON_ERROR_HANDLING_HPP

#include <boost/math/policies/error_handling.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4702) 
#endif

namespace boost{ namespace math{ namespace detail
{

template <class RealType, class Policy>
inline bool check_probability(const char* function, RealType const& prob, RealType* result, const Policy& pol)
{
if((prob < 0) || (prob > 1) || !(boost::math::isfinite)(prob))
{
*result = policies::raise_domain_error<RealType>(
function,
"Probability argument is %1%, but must be >= 0 and <= 1 !", prob, pol);
return false;
}
return true;
}

template <class RealType, class Policy>
inline bool check_df(const char* function, RealType const& df, RealType* result, const Policy& pol)
{ 
if((df <= 0) || !(boost::math::isfinite)(df))
{
*result = policies::raise_domain_error<RealType>(
function,
"Degrees of freedom argument is %1%, but must be > 0 !", df, pol);
return false;
}
return true;
}

template <class RealType, class Policy>
inline bool check_df_gt0_to_inf(const char* function, RealType const& df, RealType* result, const Policy& pol)
{  
if( (df <= 0) || (boost::math::isnan)(df) )
{ 
*result = policies::raise_domain_error<RealType>(
function,
"Degrees of freedom argument is %1%, but must be > 0 !", df, pol);
return false;
}
return true;
} 


template <class RealType, class Policy>
inline bool check_scale(
const char* function,
RealType scale,
RealType* result,
const Policy& pol)
{
if((scale <= 0) || !(boost::math::isfinite)(scale))
{ 
*result = policies::raise_domain_error<RealType>(
function,
"Scale parameter is %1%, but must be > 0 !", scale, pol);
return false;
}
return true;
}

template <class RealType, class Policy>
inline bool check_location(
const char* function,
RealType location,
RealType* result,
const Policy& pol)
{
if(!(boost::math::isfinite)(location))
{
*result = policies::raise_domain_error<RealType>(
function,
"Location parameter is %1%, but must be finite!", location, pol);
return false;
}
return true;
}

template <class RealType, class Policy>
inline bool check_x(
const char* function,
RealType x,
RealType* result,
const Policy& pol)
{
if(!(boost::math::isfinite)(x))
{
*result = policies::raise_domain_error<RealType>(
function,
"Random variate x is %1%, but must be finite!", x, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_x_not_NaN(
const char* function,
RealType x,
RealType* result,
const Policy& pol)
{
if ((boost::math::isnan)(x))
{
*result = policies::raise_domain_error<RealType>(
function,
"Random variate x is %1%, but must be finite or + or - infinity!", x, pol);
return false;
}
return true;
} 

template <class RealType, class Policy>
inline bool check_x_gt0(
const char* function,
RealType x,
RealType* result,
const Policy& pol)
{
if(x <= 0)
{
*result = policies::raise_domain_error<RealType>(
function,
"Random variate x is %1%, but must be > 0!", x, pol);
return false;
}

return true;
} 

template <class RealType, class Policy>
inline bool check_positive_x(
const char* function,
RealType x,
RealType* result,
const Policy& pol)
{
if(!(boost::math::isfinite)(x) || (x < 0))
{
*result = policies::raise_domain_error<RealType>(
function,
"Random variate x is %1%, but must be finite and >= 0!", x, pol);
return false;
}
return true;
}

template <class RealType, class Policy>
inline bool check_non_centrality(
const char* function,
RealType ncp,
RealType* result,
const Policy& pol)
{
if((ncp < 0) || !(boost::math::isfinite)(ncp))
{ 
*result = policies::raise_domain_error<RealType>(
function,
"Non centrality parameter is %1%, but must be > 0 !", ncp, pol);
return false;
}
return true;
}

template <class RealType, class Policy>
inline bool check_finite(
const char* function,
RealType x,
RealType* result,
const Policy& pol)
{
if(!(boost::math::isfinite)(x))
{ 
*result = policies::raise_domain_error<RealType>(
function,
"Parameter is %1%, but must be finite !", x, pol);
return false;
}
return true;
}

} 
} 
} 

#ifdef BOOST_MSVC
#  pragma warning(pop)
#endif

#endif 
