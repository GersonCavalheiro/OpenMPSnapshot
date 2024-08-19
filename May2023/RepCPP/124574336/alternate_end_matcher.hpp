
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ALTERNATE_END_MATCHER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_MATCHER_ALTERNATE_END_MATCHER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
# pragma warning(push)
# pragma warning(disable : 4100) 
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/core/quant_style.hpp>
#include <boost/xpressive/detail/core/state.hpp>

namespace boost { namespace xpressive { namespace detail
{

struct alternate_end_matcher
: quant_style_assertion
{
mutable void const *back_;

alternate_end_matcher()
: back_(0)
{
}

template<typename BidiIter, typename Next>
bool match(match_state<BidiIter> &state, Next const &next) const
{
return next.pop_match(state, this->back_);
}
};

}}}

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
