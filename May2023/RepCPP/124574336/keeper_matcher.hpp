
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_KEEPER_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_KEEPER_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Xpr>
struct keeper_matcher
: quant_style<quant_variable_width, unknown_width::value, Xpr::pure>
{
keeper_matcher(Xpr const &xpr, bool pure = Xpr::pure)
: xpr_(xpr)
, pure_(pure)
{
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

if(!this->xpr_.match(state))
{
return false;
}
else if(next.match(state))
{
return true;
}

state.cur_ = tmp;
return false;
}

template<typename BidiIter, typename Next>
bool match_(match_state<BidiIter> &state, Next const &next, mpl::false_) const
{
BidiIter const tmp = state.cur_;

memento<BidiIter> mem = save_sub_matches(state);

if(!this->xpr_.match(state))
{
restore_action_queue(mem, state);
reclaim_sub_matches(mem, state, false);
return false;
}
restore_action_queue(mem, state);
if(next.match(state))
{
reclaim_sub_matches(mem, state, true);
return true;
}

restore_sub_matches(mem, state);
state.cur_ = tmp;
return false;
}

Xpr xpr_;
bool pure_; 
};

}}}

#endif
