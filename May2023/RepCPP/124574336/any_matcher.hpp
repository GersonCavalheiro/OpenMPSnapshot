
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ANY_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ANY_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct any_matcher
{
BOOST_XPR_QUANT_STYLE(quant_fixed_width, 1, true)

template<typename BidiIter, typename Next>
static bool match(match_state<BidiIter> &state, Next const &next)
{
if(state.eos())
{
return false;
}

++state.cur_;
if(next.match(state))
{
return true;
}

--state.cur_;
return false;
}
};

}}}

#endif
