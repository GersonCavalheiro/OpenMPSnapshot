
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_CHARSET_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_CHARSET_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Traits, typename ICase, typename CharSet>
struct charset_matcher
: quant_style_fixed_width<1>
{
typedef typename Traits::char_type char_type;
typedef Traits traits_type;
typedef ICase icase_type;

charset_matcher(CharSet const &charset = CharSet())
: charset_(charset)
{
}

void inverse()
{
this->charset_.inverse();
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
if(state.eos() || !this->charset_.test(*state.cur_, traits_cast<Traits>(state), icase_type()))
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

CharSet charset_;
};

}}}

#endif
