
#ifndef BOOST_ACCUMULATORS_STATISTICS_WEIGHTED_TAIL_QUANTILE_HPP_DE_01_01_2006
#define BOOST_ACCUMULATORS_STATISTICS_WEIGHTED_TAIL_QUANTILE_HPP_DE_01_01_2006

#include <vector>
#include <limits>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <boost/parameter/keyword.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/depends_on.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
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

template<typename Sample, typename Weight, typename LeftRight>
struct weighted_tail_quantile_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Weight, std::size_t>::result_type float_type;
typedef Sample result_type;

weighted_tail_quantile_impl(dont_care) {}

template<typename Args>
result_type result(Args const &args) const
{
float_type threshold = sum_of_weights(args)
* ( ( is_same<LeftRight, left>::value ) ? args[quantile_probability] : 1. - args[quantile_probability] );

std::size_t n = 0;
Weight sum = Weight(0);

while (sum < threshold)
{
if (n < static_cast<std::size_t>(tail_weights(args).size()))
{
sum += *(tail_weights(args).begin() + n);
n++;
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

return *(boost::begin(tail(args)) + n - 1);
}
};
} 

namespace tag
{
template<typename LeftRight>
struct weighted_tail_quantile
: depends_on<sum_of_weights, tail_weights<LeftRight> >
{
typedef accumulators::impl::weighted_tail_quantile_impl<mpl::_1, mpl::_2, LeftRight> impl;
};
}

namespace extract
{
extractor<tag::quantile> const weighted_tail_quantile = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(weighted_tail_quantile)
}

using extract::weighted_tail_quantile;

}} 

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
