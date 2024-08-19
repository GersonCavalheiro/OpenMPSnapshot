
#if !defined(BOOST_SPIRIT_LEX_IN_STATE_OCT_09_2007_0748PM)
#define BOOST_SPIRIT_LEX_IN_STATE_OCT_09_2007_0748PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/proto/traits.hpp>

namespace boost { namespace spirit { namespace qi
{
template <typename Skipper, typename String = char const*>
struct in_state_skipper
: proto::subscript<
typename proto::terminal<
terminal_ex<tag::in_state, fusion::vector1<String> > 
>::type
, Skipper
>::type {};
}}}

#endif
