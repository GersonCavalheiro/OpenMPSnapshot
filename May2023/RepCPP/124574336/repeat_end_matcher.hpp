
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_REPEAT_END_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_REPEAT_END_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Greedy>
struct repeat_end_matcher
: quant_style<quant_none, 0, false>
{
typedef Greedy greedy_type;
int mark_number_;
unsigned int min_, max_;
mutable void const *back_;

repeat_end_matcher(int mark_nbr, unsigned int min, unsigned int max)
: mark_number_(mark_nbr)
, min_(min)
, max_(max)
, back_(0)
{
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
sub_match_impl<BidiIter> &br = state.sub_match(this->mark_number_);

if(br.zero_width_ && br.begin_ == state.cur_)
{
return next.skip_match(state);
}

bool old_zero_width = br.zero_width_;
br.zero_width_ = (br.begin_ == state.cur_);

if(this->match_(state, next, greedy_type()))
{
return true;
}

br.zero_width_ = old_zero_width;
return false;
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::true_) const
{
sub_match_impl<BidiIter> &br = state.sub_match(this->mark_number_);

if(this->max_ > br.repeat_count_)
{
++br.repeat_count_;
if(next.top_match(state, this->back_))
{
return true;
}
else if(--br.repeat_count_ < this->min_)
{
return false;
}
}

return next.skip_match(state);
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::false_) const
{
sub_match_impl<BidiIter> &br = state.sub_match(this->mark_number_);

if(this->min_ <= br.repeat_count_)
{
if(next.skip_match(state))
{
return true;
}
}

if(this->max_ > br.repeat_count_)
{
++br.repeat_count_;
if(next.top_match(state, this->back_))
{
return true;
}
--br.repeat_count_;
}

return false;
}
};

}}}

#endif
