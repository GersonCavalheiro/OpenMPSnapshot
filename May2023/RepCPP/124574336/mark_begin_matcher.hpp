
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_MARK_BEGIN_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_MARK_BEGIN_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct mark_begin_matcher
: quant_style<quant_fixed_width, 0, false>
{
int mark_number_; 

mark_begin_matcher(int mark_number)
: mark_number_(mark_number)
{
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
sub_match_impl<BidiIter> &br = state.sub_match(this->mark_number_);

BidiIter old_begin = br.begin_;
br.begin_ = state.cur_;

if(next.match(state))
{
return true;
}

br.begin_ = old_begin;
return false;
}
};

}}}

#endif
