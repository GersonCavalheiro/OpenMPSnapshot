
#ifndef BOOST_XPRESSIVE_DETAIL_CORE_SUB_MATCH_IMPL_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_CORE_SUB_MATCH_IMPL_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/xpressive/sub_match.hpp>

namespace boost { namespace xpressive { namespace detail
{


template<typename BidiIter>
struct sub_match_impl
: sub_match<BidiIter>
{
unsigned int repeat_count_;
BidiIter begin_;
bool zero_width_;

sub_match_impl(BidiIter const &begin)
: sub_match<BidiIter>(begin, begin)
, repeat_count_(0)
, begin_(begin)
, zero_width_(false)
{
}
};

}}} 

#endif
