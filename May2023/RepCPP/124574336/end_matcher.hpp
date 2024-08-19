
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_END_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_END_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/core/sub_match_impl.hpp>
#include <boost/xpressive/detail/core/flow_control.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct end_matcher
: quant_style_assertion
{
template<typename BidiIter, typename Next>
static bool match(match_state<BidiIter> &state, Next const &)
{
BidiIter const tmp = state.cur_;
sub_match_impl<BidiIter> &s0 = state.sub_match(0);
BOOST_ASSERT(!s0.matched);

if(0 != state.context_.prev_context_)
{
if(!pop_context_match(state))
{
return false;
}

s0.first = s0.begin_;
s0.second = tmp;
s0.matched = true;

return true;
}
else if((state.flags_.match_all_ && !state.eos()) ||
(state.flags_.match_not_null_ && state.cur_ == s0.begin_))
{
return false;
}

s0.first = s0.begin_;
s0.second = tmp;
s0.matched = true;

for(actionable const *actor = state.action_list_.next; 0 != actor; actor = actor->next)
{
actor->execute(state.action_args_);
}

return true;
}
};

struct independent_end_matcher
: quant_style_assertion
{
template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &) const
{
for(actionable const *actor = state.action_list_.next; 0 != actor; actor = actor->next)
{
actor->execute(state.action_args_);
}

return true;
}
};

}}}

#endif
