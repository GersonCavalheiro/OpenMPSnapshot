
#ifndef BOOST_ACCUMULATORS_STATISTICS_WEIGHTED_TAIL_VARIATE_MEANS_HPP_DE_01_01_2006
#define BOOST_ACCUMULATORS_STATISTICS_WEIGHTED_TAIL_VARIATE_MEANS_HPP_DE_01_01_2006

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
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
#include <boost/accumulators/statistics/tail.hpp>
#include <boost/accumulators/statistics/tail_variate.hpp>
#include <boost/accumulators/statistics/tail_variate_means.hpp>
#include <boost/accumulators/statistics/weighted_tail_mean.hpp>
#include <boost/accumulators/statistics/parameters/quantile_probability.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4127) 
#endif

namespace boost
{
namespace numeric { namespace functional
{
template<typename T, typename U>
struct multiply_and_promote_to_double
: multiplies<T, double const>
{
};
}}
}

namespace boost { namespace accumulators
{

namespace impl
{


template<typename Sample, typename Weight, typename Impl, typename LeftRight, typename VariateType>
struct weighted_tail_variate_means_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Weight, Weight>::result_type float_type;
typedef typename numeric::functional::fdiv<typename numeric::functional::multiplies<VariateType, Weight>::result_type, Weight>::result_type array_type;
typedef iterator_range<typename array_type::iterator> result_type;

weighted_tail_variate_means_impl(dont_care) {}

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
if (std::numeric_limits<float_type>::has_quiet_NaN)
{
std::fill(
this->tail_means_.begin()
, this->tail_means_.end()
, std::numeric_limits<float_type>::quiet_NaN()
);
}
else
{
std::ostringstream msg;
msg << "index n = " << n << " is not in valid range [0, " << tail(args).size() << ")";
boost::throw_exception(std::runtime_error(msg.str()));
}
}
}

std::size_t num_variates = tail_variate(args).begin()->size();

this->tail_means_.clear();
this->tail_means_.resize(num_variates, Sample(0));

this->tail_means_ = std::inner_product(
tail_variate(args).begin()
, tail_variate(args).begin() + n
, tail_weights(args).begin()
, this->tail_means_
, numeric::functional::plus<array_type const, array_type const>()
, numeric::functional::multiply_and_promote_to_double<VariateType const, Weight const>()
);

float_type factor = sum * ( (is_same<Impl, relative>::value) ? non_coherent_weighted_tail_mean(args) : 1. );

std::transform(
this->tail_means_.begin()
, this->tail_means_.end()
, this->tail_means_.begin()
#ifdef BOOST_NO_CXX98_BINDERS
, std::bind(numeric::functional::divides<typename array_type::value_type const, float_type const>(), std::placeholders::_1, factor)
#else
, std::bind2nd(numeric::functional::divides<typename array_type::value_type const, float_type const>(), factor)
#endif
);

return make_iterator_range(this->tail_means_);
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
ar & tail_means_;
}

private:

mutable array_type tail_means_;

};

} 

namespace tag
{
template<typename LeftRight, typename VariateType, typename VariateTag>
struct absolute_weighted_tail_variate_means
: depends_on<non_coherent_weighted_tail_mean<LeftRight>, tail_variate<VariateType, VariateTag, LeftRight>, tail_weights<LeftRight> >
{
typedef accumulators::impl::weighted_tail_variate_means_impl<mpl::_1, mpl::_2, absolute, LeftRight, VariateType> impl;
};
template<typename LeftRight, typename VariateType, typename VariateTag>
struct relative_weighted_tail_variate_means
: depends_on<non_coherent_weighted_tail_mean<LeftRight>, tail_variate<VariateType, VariateTag, LeftRight>, tail_weights<LeftRight> >
{
typedef accumulators::impl::weighted_tail_variate_means_impl<mpl::_1, mpl::_2, relative, LeftRight, VariateType> impl;
};
}

namespace extract
{
extractor<tag::abstract_absolute_tail_variate_means> const weighted_tail_variate_means = {};
extractor<tag::abstract_relative_tail_variate_means> const relative_weighted_tail_variate_means = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(weighted_tail_variate_means)
BOOST_ACCUMULATORS_IGNORE_GLOBAL(relative_weighted_tail_variate_means)
}

using extract::weighted_tail_variate_means;
using extract::relative_weighted_tail_variate_means;

template<typename LeftRight, typename VariateType, typename VariateTag>
struct as_feature<tag::weighted_tail_variate_means<LeftRight, VariateType, VariateTag>(absolute)>
{
typedef tag::absolute_weighted_tail_variate_means<LeftRight, VariateType, VariateTag> type;
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct as_feature<tag::weighted_tail_variate_means<LeftRight, VariateType, VariateTag>(relative)>
{
typedef tag::relative_weighted_tail_variate_means<LeftRight, VariateType, VariateTag> type;
};

}} 

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
