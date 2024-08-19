
#ifndef BOOST_ACCUMULATORS_STATISTICS_PEAKS_OVER_THRESHOLD_HPP_DE_01_01_2006
#define BOOST_ACCUMULATORS_STATISTICS_PEAKS_OVER_THRESHOLD_HPP_DE_01_01_2006

#include <vector>
#include <limits>
#include <numeric>
#include <functional>
#include <boost/config/no_tr1/cmath.hpp> 
#include <sstream> 
#include <stdexcept> 
#include <boost/throw_exception.hpp>
#include <boost/range.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/int.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/parameter/keyword.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/accumulators/accumulators_fwd.hpp>
#include <boost/accumulators/framework/accumulator_base.hpp>
#include <boost/accumulators/framework/extractor.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/framework/parameters/sample.hpp>
#include <boost/accumulators/framework/depends_on.hpp>
#include <boost/accumulators/statistics_fwd.hpp>
#include <boost/accumulators/statistics/parameters/quantile_probability.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/tail.hpp>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable: 4127) 
#endif

namespace boost { namespace accumulators
{

BOOST_PARAMETER_NESTED_KEYWORD(tag, pot_threshold_value, threshold_value)
BOOST_PARAMETER_NESTED_KEYWORD(tag, pot_threshold_probability, threshold_probability)

BOOST_ACCUMULATORS_IGNORE_GLOBAL(pot_threshold_value)
BOOST_ACCUMULATORS_IGNORE_GLOBAL(pot_threshold_probability)

namespace impl
{

template<typename Sample, typename LeftRight>
struct peaks_over_threshold_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Sample, std::size_t>::result_type float_type;
typedef boost::tuple<float_type, float_type, float_type> result_type;
typedef mpl::int_<is_same<LeftRight, left>::value ? -1 : 1> sign;

template<typename Args>
peaks_over_threshold_impl(Args const &args)
: Nu_(0)
, mu_(sign::value * numeric::fdiv(args[sample | Sample()], (std::size_t)1))
, sigma2_(numeric::fdiv(args[sample | Sample()], (std::size_t)1))
, threshold_(sign::value * args[pot_threshold_value])
, fit_parameters_(boost::make_tuple(0., 0., 0.))
, is_dirty_(true)
{
}

template<typename Args>
void operator ()(Args const &args)
{
this->is_dirty_ = true;

if (sign::value * args[sample] > this->threshold_)
{
this->mu_ += args[sample];
this->sigma2_ += args[sample] * args[sample];
++this->Nu_;
}
}

template<typename Args>
result_type result(Args const &args) const
{
if (this->is_dirty_)
{
this->is_dirty_ = false;

std::size_t cnt = count(args);

this->mu_ = sign::value * numeric::fdiv(this->mu_, this->Nu_);
this->sigma2_ = numeric::fdiv(this->sigma2_, this->Nu_);
this->sigma2_ -= this->mu_ * this->mu_;

float_type threshold_probability = numeric::fdiv(cnt - this->Nu_, cnt);

float_type tmp = numeric::fdiv(( this->mu_ - this->threshold_ )*( this->mu_ - this->threshold_ ), this->sigma2_);
float_type xi_hat = 0.5 * ( 1. - tmp );
float_type beta_hat = 0.5 * ( this->mu_ - this->threshold_ ) * ( 1. + tmp );
float_type beta_bar = beta_hat * std::pow(1. - threshold_probability, xi_hat);
float_type u_bar = this->threshold_ - beta_bar * ( std::pow(1. - threshold_probability, -xi_hat) - 1.)/xi_hat;
this->fit_parameters_ = boost::make_tuple(u_bar, beta_bar, xi_hat);
}

return this->fit_parameters_;
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{ 
ar & Nu_;
ar & mu_;
ar & sigma2_;
ar & threshold_;
ar & get<0>(fit_parameters_);
ar & get<1>(fit_parameters_);
ar & get<2>(fit_parameters_);
ar & is_dirty_;
}

private:
std::size_t Nu_;                     
mutable float_type mu_;              
mutable float_type sigma2_;          
float_type threshold_;
mutable result_type fit_parameters_; 
mutable bool is_dirty_;
};


template<typename Sample, typename LeftRight>
struct peaks_over_threshold_prob_impl
: accumulator_base
{
typedef typename numeric::functional::fdiv<Sample, std::size_t>::result_type float_type;
typedef boost::tuple<float_type, float_type, float_type> result_type;
typedef mpl::int_<is_same<LeftRight, left>::value ? -1 : 1> sign;

template<typename Args>
peaks_over_threshold_prob_impl(Args const &args)
: mu_(sign::value * numeric::fdiv(args[sample | Sample()], (std::size_t)1))
, sigma2_(numeric::fdiv(args[sample | Sample()], (std::size_t)1))
, threshold_probability_(args[pot_threshold_probability])
, fit_parameters_(boost::make_tuple(0., 0., 0.))
, is_dirty_(true)
{
}

void operator ()(dont_care)
{
this->is_dirty_ = true;
}

template<typename Args>
result_type result(Args const &args) const
{
if (this->is_dirty_)
{
this->is_dirty_ = false;

std::size_t cnt = count(args);

std::size_t n = static_cast<std::size_t>(
std::ceil(
cnt * ( ( is_same<LeftRight, left>::value ) ? this->threshold_probability_ : 1. - this->threshold_probability_ )
)
);

if ( n >= static_cast<std::size_t>(tail(args).size()))
{
if (std::numeric_limits<float_type>::has_quiet_NaN)
{
return boost::make_tuple(
std::numeric_limits<float_type>::quiet_NaN()
, std::numeric_limits<float_type>::quiet_NaN()
, std::numeric_limits<float_type>::quiet_NaN()
);
}
else
{
std::ostringstream msg;
msg << "index n = " << n << " is not in valid range [0, " << tail(args).size() << ")";
boost::throw_exception(std::runtime_error(msg.str()));
return boost::make_tuple(Sample(0), Sample(0), Sample(0));
}
}
else
{
float_type u = *(tail(args).begin() + n - 1) * sign::value;

for (std::size_t i = 0; i < n; ++i)
{
mu_ += *(tail(args).begin() + i);
sigma2_ += *(tail(args).begin() + i) * (*(tail(args).begin() + i));
}

this->mu_ = sign::value * numeric::fdiv(this->mu_, n);
this->sigma2_ = numeric::fdiv(this->sigma2_, n);
this->sigma2_ -= this->mu_ * this->mu_;

if (is_same<LeftRight, left>::value)
this->threshold_probability_ = 1. - this->threshold_probability_;

float_type tmp = numeric::fdiv(( this->mu_ - u )*( this->mu_ - u ), this->sigma2_);
float_type xi_hat = 0.5 * ( 1. - tmp );
float_type beta_hat = 0.5 * ( this->mu_ - u ) * ( 1. + tmp );
float_type beta_bar = beta_hat * std::pow(1. - threshold_probability_, xi_hat);
float_type u_bar = u - beta_bar * ( std::pow(1. - threshold_probability_, -xi_hat) - 1.)/xi_hat;
this->fit_parameters_ = boost::make_tuple(u_bar, beta_bar, xi_hat);
}
}

return this->fit_parameters_;
}

template<class Archive>
void serialize(Archive & ar, const unsigned int file_version)
{ 
ar & mu_;
ar & sigma2_;
ar & threshold_probability_;
ar & get<0>(fit_parameters_);
ar & get<1>(fit_parameters_);
ar & get<2>(fit_parameters_);
ar & is_dirty_;
}

private:
mutable float_type mu_;                     
mutable float_type sigma2_;                 
mutable float_type threshold_probability_;
mutable result_type fit_parameters_;        
mutable bool is_dirty_;
};

} 

namespace tag
{
template<typename LeftRight>
struct peaks_over_threshold
: depends_on<count>
, pot_threshold_value
{
typedef accumulators::impl::peaks_over_threshold_impl<mpl::_1, LeftRight> impl;
};

template<typename LeftRight>
struct peaks_over_threshold_prob
: depends_on<count, tail<LeftRight> >
, pot_threshold_probability
{
typedef accumulators::impl::peaks_over_threshold_prob_impl<mpl::_1, LeftRight> impl;
};

struct abstract_peaks_over_threshold
: depends_on<>
{
};
}

namespace extract
{
extractor<tag::abstract_peaks_over_threshold> const peaks_over_threshold = {};

BOOST_ACCUMULATORS_IGNORE_GLOBAL(peaks_over_threshold)
}

using extract::peaks_over_threshold;

template<typename LeftRight>
struct as_feature<tag::peaks_over_threshold<LeftRight>(with_threshold_value)>
{
typedef tag::peaks_over_threshold<LeftRight> type;
};

template<typename LeftRight>
struct as_feature<tag::peaks_over_threshold<LeftRight>(with_threshold_probability)>
{
typedef tag::peaks_over_threshold_prob<LeftRight> type;
};

template<typename LeftRight>
struct feature_of<tag::peaks_over_threshold<LeftRight> >
: feature_of<tag::abstract_peaks_over_threshold>
{
};

template<typename LeftRight>
struct feature_of<tag::peaks_over_threshold_prob<LeftRight> >
: feature_of<tag::abstract_peaks_over_threshold>
{
};

template<typename LeftRight>
struct as_weighted_feature<tag::peaks_over_threshold<LeftRight> >
{
typedef tag::weighted_peaks_over_threshold<LeftRight> type;
};

template<typename LeftRight>
struct feature_of<tag::weighted_peaks_over_threshold<LeftRight> >
: feature_of<tag::peaks_over_threshold<LeftRight> >
{};

template<typename LeftRight>
struct as_weighted_feature<tag::peaks_over_threshold_prob<LeftRight> >
{
typedef tag::weighted_peaks_over_threshold_prob<LeftRight> type;
};

template<typename LeftRight>
struct feature_of<tag::weighted_peaks_over_threshold_prob<LeftRight> >
: feature_of<tag::peaks_over_threshold_prob<LeftRight> >
{};

}} 

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
