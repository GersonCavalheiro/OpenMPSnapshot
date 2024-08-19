
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ASSERT_EOL_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ASSERT_EOL_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/next_prior.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/core/matcher/assert_line_base.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Traits>
struct assert_eol_matcher
: assert_line_base<Traits>
{
typedef typename Traits::char_type char_type;

assert_eol_matcher(Traits const &tr)
: assert_line_base<Traits>(tr)
{
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
if(state.eos())
{
if(!state.flags_.match_eol_)
{
return false;
}
}
else
{
char_type ch = *state.cur_;

if(!traits_cast<Traits>(state).isctype(ch, this->newline_))
{
return false;
}
else if(ch == this->nl_ && (!state.bos() || state.flags_.match_prev_avail_) && *boost::prior(state.cur_) == this->cr_)
{
return false;
}
}

return next.match(state);
}
};

}}}

#endif
