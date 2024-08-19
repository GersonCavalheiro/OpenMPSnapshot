
#ifndef BOOST_ACCUMULATORS_STATISTICS_TAIL_MEAN_HPP_DE_01_01_2006
#define BOOST_ACCUMULATORS_STATISTICS_TAIL_MEAN_HPP_DE_01_01_2006

#include <numeric>
#include <vector>
#include <limits>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <boost/parameter/keyword.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/tail.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics/parameters/quantile_probability.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4127) 
#endif

namespace boost { namespace accumulators
{

namespace impl
{


template<typename Sample, typename LeftRight>
struct coherent_tail_mean_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Sample, std::size_t>::result_type float_type;
typedef float_type result_type;

coherent_tail_mean_impl(dont_care) {}

template<typename Args>
result_type result(Args const &args) const
{
std::size_t cnt = count(args);

std::size_t n = static_cast<std::size_t>(
std::ceil(
cnt * ( ( is_same<LeftRight, left>::value ) ? args[quantile_probability] : 1. - args[quantile_probability] )
)
);

extractor<tag::non_coherent_tail_mean<LeftRight> > const some_non_coherent_tail_mean = {};

return some_non_coherent_tail_mean(args)
+ numeric::fdiv(quantile(args), n)
* (
( is_same<LeftRight, left>::value ) ? args[quantile_probability] : 1. - args[quantile_probability]
- numeric::fdiv(n, count(args))
);
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version) {}
};


template<typename Sample, typename LeftRight>
struct non_coherent_tail_mean_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Sample, std::size_t>::result_type float_type;
typedef float_type result_type;

non_coherent_tail_mean_impl(dont_care) {}

template<typename Args>
result_type result(Args const &args) const
{
std::size_t cnt = count(args);

std::size_t n = static_cast<std::size_t>(
std::ceil(
cnt * ( ( is_same<LeftRight, left>::value ) ? args[quantile_probability] : 1. - args[quantile_probability] )
)
);

if (n <= static_cast<std::size_t>(tail(args).size()))
return numeric::fdiv(
std::accumulate(
tail(args).begin()
, tail(args).begin() + n
, Sample(0)
)
, n
);
else
{
if (std::numeric_limits<result_type>::has_quiet_NaN)
{
return std::numeric_limits<result_type>::quiet_NaN();
}
else
{
std::ostringstream msg;
msg << "index n = " << n << " is not in valid range [0, " << tail(args).size() << ")";
boost::throw_exception(std::runtime_error(msg.str()));
return Sample(0);
}
}
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version) {}
};

} 


namespace tag
{
template<typename LeftRight>
struct coherent_tail_mean
: depends_on<count, quantile, non_coherent_tail_mean<LeftRight> >
{
typedef accumulators::impl::coherent_tail_mean_impl<mpl::_1, LeftRight> impl;
};

template<typename LeftRight>
struct non_coherent_tail_mean
: depends_on<count, tail<LeftRight> >
{
typedef accumulators::impl::non_coherent_tail_mean_impl<mpl::_1, LeftRight> impl;
};

struct abstract_non_coherent_tail_mean
: depends_on<>
{
};
}

namespace extract
{
extractor<tag::abstract_non_coherent_tail_mean> const non_coherent_tail_mean = {};
extractor<tag::tail_mean> const coherent_tail_mean = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(non_coherent_tail_mean)
BOOST_ACCUMULATORS_IGNORE_GLOBAL(coherent_tail_mean)
}

using extract::non_coherent_tail_mean;
using extract::coherent_tail_mean;

template<typename LeftRight>
struct feature_of<tag::coherent_tail_mean<LeftRight> >
: feature_of<tag::tail_mean>
{
};

template<typename LeftRight>
struct feature_of<tag::non_coherent_tail_mean<LeftRight> >
: feature_of<tag::abstract_non_coherent_tail_mean>
{
};

template<typename LeftRight>
struct as_weighted_feature<tag::non_coherent_tail_mean<LeftRight> >
{
typedef tag::non_coherent_weighted_tail_mean<LeftRight> type;
};

template<typename LeftRight>
struct feature_of<tag::non_coherent_weighted_tail_mean<LeftRight> >
: feature_of<tag::non_coherent_tail_mean<LeftRight> >
{};


}} 

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
