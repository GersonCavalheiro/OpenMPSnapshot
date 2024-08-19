
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_LOOKAHEAD_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_LOOKAHEAD_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/utility/save_restore.hpp>
#include <boost/xpressive/detail/utility/ignore_unused.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Xpr>
struct lookahead_matcher
: quant_style<quant_none, 0, Xpr::pure>
{
lookahead_matcher(Xpr const &xpr, bool no, bool pure = Xpr::pure)
: xpr_(xpr)
, not_(no)
, pure_(pure)
{
}

void inverse()
{
this->not_ = !this->not_;
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
return Xpr::pure || this->pure_
? this->match_(state, next, mpl::true_())
: this->match_(state, next, mpl::false_());
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::true_) const
{
BidiIter const tmp = state.cur_;

if(this->not_)
{
save_restore<bool> partial_match(state.found_partial_match_);
detail::ignore_unused(partial_match);

if(this->xpr_.match(state))
{
state.cur_ = tmp;
return false;
}
else if(next.match(state))
{
return true;
}
}
else
{
if(!this->xpr_.match(state))
{
return false;
}
state.cur_ = tmp;
if(next.match(state))
{
return true;
}
}

BOOST_ASSERT(state.cur_ == tmp);
return false;
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::false_) const
{
BidiIter const tmp = state.cur_;

memento<BidiIter> mem = save_sub_matches(state);

if(this->not_)
{
save_restore<bool> partial_match(state.found_partial_match_);
detail::ignore_unused(partial_match);

if(this->xpr_.match(state))
{
restore_action_queue(mem, state);
restore_sub_matches(mem, state);
state.cur_ = tmp;
return false;
}
restore_action_queue(mem, state);
if(next.match(state))
{
reclaim_sub_matches(mem, state, true);
return true;
}
reclaim_sub_matches(mem, state, false);
}
else
{
if(!this->xpr_.match(state))
{
restore_action_queue(mem, state);
reclaim_sub_matches(mem, state, false);
return false;
}
state.cur_ = tmp;
restore_action_queue(mem, state);
if(next.match(state))
{
reclaim_sub_matches(mem, state, true);
return true;
}
restore_sub_matches(mem, state);
}

BOOST_ASSERT(state.cur_ == tmp);
return false;
}

Xpr xpr_;
bool not_;
bool pure_; 
};

}}}

#endif
