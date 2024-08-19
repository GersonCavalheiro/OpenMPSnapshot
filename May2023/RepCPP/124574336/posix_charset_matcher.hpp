
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_POSIX_CHARSET_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_POSIX_CHARSET_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>
#include <boost/xpressive/detail/utility/traits_utils.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Traits>
struct posix_charset_matcher
: quant_style_fixed_width<1>
{
typedef Traits traits_type;
typedef typename Traits::char_class_type char_class_type;

posix_charset_matcher(char_class_type m, bool no)
: not_(no)
, mask_(m)
{
BOOST_ASSERT(0 != this->mask_);
}

void inverse()
{
this->not_ = !this->not_;
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
if(state.eos() || this->not_ == traits_cast<Traits>(state).isctype(
*state.cur_, this->mask_))
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

bool not_;
char_class_type mask_;
};

}}}

#endif
