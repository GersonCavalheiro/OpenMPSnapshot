
#if !defined(BOOST_SPIRIT_KARMA_DELIMIT_FEB_20_2007_1208PM)
#define BOOST_SPIRIT_KARMA_DELIMIT_FEB_20_2007_1208PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/karma/detail/unused_delimiter.hpp>

namespace boost { namespace spirit { namespace karma
{
template <typename OutputIterator, typename Delimiter>
inline bool delimit_out(OutputIterator& sink, Delimiter const& d)
{
return d.generate(sink, unused, unused, unused);
}

template <typename OutputIterator>
inline bool delimit_out(OutputIterator&, unused_type)
{
return true;
}

template <typename OutputIterator, typename Delimiter>
inline bool delimit_out(OutputIterator&
, detail::unused_delimiter<Delimiter> const&)
{
return true;
}

}}}

#endif

