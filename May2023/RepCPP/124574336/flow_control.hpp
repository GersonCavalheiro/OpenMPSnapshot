
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_FLOW_CONTROL_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_FLOW_CONTROL_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/regex_impl.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/utility/ignore_unused.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename BidiIter>
inline bool push_context_match
(
regex_impl<BidiIter> const &impl
, match_state<BidiIter> &state
, matchable<BidiIter> const &next
)
{
if(state.is_active_regex(impl) && state.cur_ == state.sub_match(0).begin_)
{
return next.match(state);
}

match_context<BidiIter> context = state.push_context(impl, next, context);
detail::ignore_unused(context);

return state.pop_context(impl, impl.xpr_->match(state));
}

template<typename BidiIter>
inline bool pop_context_match(match_state<BidiIter> &state)
{
match_context<BidiIter> &context(*state.context_.prev_context_);
state.swap_context(context);

bool success = context.next_ptr_->match(state);

state.swap_context(context);
return success;
}

}}} 

#endif

