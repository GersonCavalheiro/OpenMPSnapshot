
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_REGEX_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_REGEX_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/assert.hpp>
#include <boost/xpressive/regex_error.hpp>
#include <boost/xpressive/regex_constants.hpp>
#include <boost/xpressive/detail/core/regex_impl.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/core/adaptor.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename BidiIter>
struct regex_matcher
: quant_style<quant_variable_width, unknown_width::value, false>
{
regex_impl<BidiIter> impl_;

regex_matcher(shared_ptr<regex_impl<BidiIter> > const &impl)
: impl_()
{
this->impl_.xpr_ = impl->xpr_;
this->impl_.traits_ = impl->traits_;
this->impl_.mark_count_ = impl->mark_count_;
this->impl_.hidden_mark_count_ = impl->hidden_mark_count_;

BOOST_XPR_ENSURE_(this->impl_.xpr_, regex_constants::error_badref, "bad regex reference");
}

template<typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
BOOST_MPL_ASSERT((is_static_xpression<Next>));

xpression_adaptor<reference_wrapper<Next const>, matchable<BidiIter> > adaptor(boost::cref(next));
return push_context_match(this->impl_, state, adaptor);
}
};

}}}

#endif
