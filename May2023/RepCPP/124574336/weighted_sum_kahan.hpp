
#ifndef BOOST_ACCUMULATORS_STATISTICS_WEIGHTED_SUM_KAHAN_HPP_EAN_11_05_2011
#define BOOST_ACCUMULATORS_STATISTICS_WEIGHTED_SUM_KAHAN_HPP_EAN_11_05_2011

#include <boost/mpl/placeholders.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/framework/parameters/weight.hpp>
#include <boost/accumulators/framework/accumulators/external_accumulator.hpp>
#include <boost/accumulators/framework/depends_on.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
#include <boost/accumulators/statistics/weighted_sum.hpp>
#include <boost/numeric/conversion/cast.hpp>

namespace boost { namespace accumulators
{

namespace impl
{
#if _MSC_VER > 1400
# pragma float_control(push)
# pragma float_control(precise, on)
#endif

template<typename Sample, typename Weight, typename Tag>
struct weighted_sum_kahan_impl
: accumulator_base
{
typedef typename numeric::functional::multiplies<Sample, Weight>::result_type weighted_sample;

typedef weighted_sample result_type;

template<typename Args>
weighted_sum_kahan_impl(Args const &args)
: weighted_sum_(
args[parameter::keyword<Tag>::get() | Sample()] * numeric::one<Weight>::value),
compensation(boost::numeric_cast<weighted_sample>(0.0))
{
}

template<typename Args>
void 
#if BOOST_ACCUMULATORS_GCC_VERSION > 40305
__attribute__((__optimize__("no-associative-math")))
#endif
operator ()(Args const &args)
{
const weighted_sample myTmp1 = args[parameter::keyword<Tag>::get()] * args[weight] - this->compensation;
const weighted_sample myTmp2 = this->weighted_sum_ + myTmp1;
this->compensation = (myTmp2 - this->weighted_sum_) - myTmp1;
this->weighted_sum_ = myTmp2;

}

result_type result(dont_care) const
{
return this->weighted_sum_;
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{
ar & weighted_sum_;
ar & compensation;
}

private:
weighted_sample weighted_sum_;
weighted_sample compensation;
};

#if _MSC_VER > 1400
# pragma float_control(pop)
#endif

} 

namespace tag
{
struct weighted_sum_kahan
: depends_on<>
{
typedef accumulators::impl::weighted_sum_kahan_impl<mpl::_1, mpl::_2, tag::sample> impl;
};

template<typename VariateType, typename VariateTag>
struct weighted_sum_of_variates_kahan
: depends_on<>
{
typedef accumulators::impl::weighted_sum_kahan_impl<VariateType, mpl::_2, VariateTag> impl;
};

}

namespace extract
{
extractor<tag::weighted_sum_kahan> const weighted_sum_kahan = {};
extractor<tag::abstract_weighted_sum_of_variates> const weighted_sum_of_variates_kahan = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(weighted_sum_kahan)
BOOST_ACCUMULATORS_IGNORE_GLOBAL(weighted_sum_of_variates_kahan)
}

using extract::weighted_sum_kahan;
using extract::weighted_sum_of_variates_kahan;

template<>
struct as_feature<tag::weighted_sum(kahan)>
{
typedef tag::weighted_sum_kahan type;
};

template<typename VariateType, typename VariateTag>
struct feature_of<tag::weighted_sum_of_variates_kahan<VariateType, VariateTag> >
: feature_of<tag::abstract_weighted_sum_of_variates>
{
};

}} 

#endif
