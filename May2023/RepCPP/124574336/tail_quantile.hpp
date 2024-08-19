
#ifndef BOOST_ACCUMULATORS_STATISTICS_TAIL_QUANTILE_HPP_DE_01_01_2006
#define BOOST_ACCUMULATORS_STATISTICS_TAIL_QUANTILE_HPP_DE_01_01_2006

#include <vector>
#include <limits>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <boost/config/no_tr1/cmath.hpp>             
#include <boost/throw_exception.hpp>
#include <boost/parameter/keyword.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/accumulators/framework/depends_on.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
#include <boost/accumulators/statistics/tail.hpp>
#include <boost/accumulators/statistics/count.hpp>
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
struct tail_quantile_impl
: accumulator_base
{
typedef Sample result_type;

tail_quantile_impl(dont_care) {}

template<typename Args>
result_type result(Args const &args) const
{
std::size_t cnt = count(args);

std::size_t n = static_cast<std::size_t>(
std::ceil(
cnt * ( ( is_same<LeftRight, left>::value ) ? args[quantile_probability] : 1. - args[quantile_probability] )
)
);

if ( n < static_cast<std::size_t>(tail(args).size()))
{
return *(boost::begin(tail(args)) + n - 1);
}
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
struct tail_quantile
: depends_on<count, tail<LeftRight> >
{
typedef accumulators::impl::tail_quantile_impl<mpl::_1, LeftRight> impl;
};
}

namespace extract
{
extractor<tag::quantile> const tail_quantile = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(tail_quantile)
}

using extract::tail_quantile;

template<typename LeftRight>
struct feature_of<tag::tail_quantile<LeftRight> >
: feature_of<tag::quantile>
{
};

template<typename LeftRight>
struct as_weighted_feature<tag::tail_quantile<LeftRight> >
{
typedef tag::weighted_tail_quantile<LeftRight> type;
};

template<typename LeftRight>
struct feature_of<tag::weighted_tail_quantile<LeftRight> >
: feature_of<tag::tail_quantile<LeftRight> >
{};

}} 

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
