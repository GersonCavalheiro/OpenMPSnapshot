
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_TRUE_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_TRUE_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct true_matcher
: quant_style_assertion
{
template<typename BidiIter, typename Next>
static bool match(match_state<BidiIter> &, Next const &)
{
return true;
}
};

}}}

#endif
