
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ATTR_BEGIN_MATCHER_HPP_EAN_06_09_2007
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ATTR_BEGIN_MATCHER_HPP_EAN_06_09_2007

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Nbr>
struct attr_begin_matcher
: quant_style<quant_none, 0, false>
{
template<typename BidiIter, typename Next>
static bool match(match_state<BidiIter> &state, Next const &next)
{
void const *attr_slots[Nbr::value] = {};
attr_context old_attr_context = state.attr_context_;
state.attr_context_.attr_slots_ = attr_slots;
state.attr_context_.prev_attr_context_ = &old_attr_context;

if(next.match(state))
{
return true;
}

state.attr_context_ = old_attr_context;
return false;
}
};

}}}

#endif
