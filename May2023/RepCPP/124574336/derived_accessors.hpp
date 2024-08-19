
#ifndef BOOST_STATS_DERIVED_HPP
#define BOOST_STATS_DERIVED_HPP



#include <boost/type_traits/is_same.hpp>
#include <boost/static_assert.hpp>

#ifdef BOOST_MSVC
# pragma warning(push)
# pragma warning(disable: 4723) 
#endif

namespace boost{ namespace math{

template <class Distribution>
typename Distribution::value_type variance(const Distribution& dist);

template <class Distribution>
inline typename Distribution::value_type standard_deviation(const Distribution& dist)
{
BOOST_MATH_STD_USING  
return sqrt(variance(dist));
}

template <class Distribution>
inline typename Distribution::value_type variance(const Distribution& dist)
{
typename Distribution::value_type result = standard_deviation(dist);
return result * result;
}

template <class Distribution, class RealType>
inline typename Distribution::value_type hazard(const Distribution& dist, const RealType& x)
{ 
typedef typename Distribution::value_type value_type;
typedef typename Distribution::policy_type policy_type;
value_type p = cdf(complement(dist, x));
value_type d = pdf(dist, x);
if(d > p * tools::max_value<value_type>())
return policies::raise_overflow_error<value_type>(
"boost::math::hazard(const Distribution&, %1%)", 0, policy_type());
if(d == 0)
{
return 0;
}
return d / p;
}

template <class Distribution, class RealType>
inline typename Distribution::value_type chf(const Distribution& dist, const RealType& x)
{ 
BOOST_MATH_STD_USING
return -log(cdf(complement(dist, x)));
}

template <class Distribution>
inline typename Distribution::value_type coefficient_of_variation(const Distribution& dist)
{
typedef typename Distribution::value_type value_type;
typedef typename Distribution::policy_type policy_type;

using std::abs;

value_type m = mean(dist);
value_type d = standard_deviation(dist);
if((abs(m) < 1) && (d > abs(m) * tools::max_value<value_type>()))
{ 
return policies::raise_overflow_error<value_type>("boost::math::coefficient_of_variation(const Distribution&, %1%)", 0, policy_type());
}
return d / m; 
}
template <class Distribution, class RealType>
inline typename Distribution::value_type pdf(const Distribution& dist, const RealType& x)
{
typedef typename Distribution::value_type value_type;
return pdf(dist, static_cast<value_type>(x));
}
template <class Distribution, class RealType>
inline typename Distribution::value_type cdf(const Distribution& dist, const RealType& x)
{
typedef typename Distribution::value_type value_type;
return cdf(dist, static_cast<value_type>(x));
}
template <class Distribution, class RealType>
inline typename Distribution::value_type quantile(const Distribution& dist, const RealType& x)
{
typedef typename Distribution::value_type value_type;
return quantile(dist, static_cast<value_type>(x));
}

template <class Distribution, class RealType>
inline typename Distribution::value_type cdf(const complemented2_type<Distribution, RealType>& c)
{
typedef typename Distribution::value_type value_type;
return cdf(complement(c.dist, static_cast<value_type>(c.param)));
}

template <class Distribution, class RealType>
inline typename Distribution::value_type quantile(const complemented2_type<Distribution, RealType>& c)
{
typedef typename Distribution::value_type value_type;
return quantile(complement(c.dist, static_cast<value_type>(c.param)));
}

template <class Dist>
inline typename Dist::value_type median(const Dist& d)
{ 
typedef typename Dist::value_type value_type;
return quantile(d, static_cast<value_type>(0.5f));
}

} 
} 


#ifdef BOOST_MSVC
# pragma warning(pop)
#endif

#endif 
