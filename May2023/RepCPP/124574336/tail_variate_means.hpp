
#ifndef BOOST_ACCUMULATORS_STATISTICS_TAIL_VARIATE_MEANS_HPP_DE_01_01_2006
#define BOOST_ACCUMULATORS_STATISTICS_TAIL_VARIATE_MEANS_HPP_DE_01_01_2006

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
#include <boost/accumulators/statistics/tail.hpp>
#include <boost/accumulators/statistics/tail_variate.hpp>
#include <boost/accumulators/statistics/tail_mean.hpp>
#include <boost/accumulators/statistics/parameters/quantile_probability.hpp>
#include <boost/serialization/vector.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4127) 
#endif

namespace boost { namespace accumulators
{

namespace impl
{


template<typename Sample, typename Impl, typename LeftRight, typename VariateTag>
struct tail_variate_means_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Sample, std::size_t>::result_type float_type;
typedef std::vector<float_type> array_type;
typedef iterator_range<typename array_type::iterator> result_type;

tail_variate_means_impl(dont_care) {}

template<typename Args>
result_type result(Args const &args) const
{
std::size_t cnt = count(args);

std::size_t n = static_cast<std::size_t>(
std::ceil(
cnt * ( ( is_same<LeftRight, left>::value ) ? args[quantile_probability] : 1. - args[quantile_probability] )
)
);

std::size_t num_variates = tail_variate(args).begin()->size();

this->tail_means_.clear();
this->tail_means_.resize(num_variates, Sample(0));

if (n < static_cast<std::size_t>(tail(args).size()))
{
this->tail_means_ = std::accumulate(
tail_variate(args).begin()
, tail_variate(args).begin() + n
, this->tail_means_
, numeric::plus
);

float_type factor = n * ( (is_same<Impl, relative>::value) ? non_coherent_tail_mean(args) : 1. );

std::transform(
this->tail_means_.begin()
, this->tail_means_.end()
, this->tail_means_.begin()
#ifdef BOOST_NO_CXX98_BINDERS
, std::bind(std::divides<float_type>(), std::placeholders::_1, factor)
#else
, std::bind2nd(std::divides<float_type>(), factor)
#endif
);
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
struct absolute_tail_variate_means
: depends_on<count, non_coherent_tail_mean<LeftRight>, tail_variate<VariateType, VariateTag, LeftRight> >
{
typedef accumulators::impl::tail_variate_means_impl<mpl::_1, absolute, LeftRight, VariateTag> impl;
};
template<typename LeftRight, typename VariateType, typename VariateTag>
struct relative_tail_variate_means
: depends_on<count, non_coherent_tail_mean<LeftRight>, tail_variate<VariateType, VariateTag, LeftRight> >
{
typedef accumulators::impl::tail_variate_means_impl<mpl::_1, relative, LeftRight, VariateTag> impl;
};
struct abstract_absolute_tail_variate_means
: depends_on<>
{
};
struct abstract_relative_tail_variate_means
: depends_on<>
{
};
}

namespace extract
{
extractor<tag::abstract_absolute_tail_variate_means> const tail_variate_means = {};
extractor<tag::abstract_relative_tail_variate_means> const relative_tail_variate_means = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(tail_variate_means)
BOOST_ACCUMULATORS_IGNORE_GLOBAL(relative_tail_variate_means)
}

using extract::tail_variate_means;
using extract::relative_tail_variate_means;

template<typename LeftRight, typename VariateType, typename VariateTag>
struct as_feature<tag::tail_variate_means<LeftRight, VariateType, VariateTag>(absolute)>
{
typedef tag::absolute_tail_variate_means<LeftRight, VariateType, VariateTag> type;
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct as_feature<tag::tail_variate_means<LeftRight, VariateType, VariateTag>(relative)>
{
typedef tag::relative_tail_variate_means<LeftRight, VariateType, VariateTag> type;
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct feature_of<tag::absolute_tail_variate_means<LeftRight, VariateType, VariateTag> >
: feature_of<tag::abstract_absolute_tail_variate_means>
{
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct feature_of<tag::relative_tail_variate_means<LeftRight, VariateType, VariateTag> >
: feature_of<tag::abstract_relative_tail_variate_means>
{
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct as_weighted_feature<tag::absolute_tail_variate_means<LeftRight, VariateType, VariateTag> >
{
typedef tag::absolute_weighted_tail_variate_means<LeftRight, VariateType, VariateTag> type;
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct feature_of<tag::absolute_weighted_tail_variate_means<LeftRight, VariateType, VariateTag> >
: feature_of<tag::absolute_tail_variate_means<LeftRight, VariateType, VariateTag> >
{
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct as_weighted_feature<tag::relative_tail_variate_means<LeftRight, VariateType, VariateTag> >
{
typedef tag::relative_weighted_tail_variate_means<LeftRight, VariateType, VariateTag> type;
};

template<typename LeftRight, typename VariateType, typename VariateTag>
struct feature_of<tag::relative_weighted_tail_variate_means<LeftRight, VariateType, VariateTag> >
: feature_of<tag::relative_tail_variate_means<LeftRight, VariateType, VariateTag> >
{
};

}} 

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
