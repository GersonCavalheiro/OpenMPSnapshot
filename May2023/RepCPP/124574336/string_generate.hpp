
#if !defined(BOOST_SPIRIT_KARMA_STRING_GENERATE_FEB_23_2007_1232PM)
#define BOOST_SPIRIT_KARMA_STRING_GENERATE_FEB_23_2007_1232PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/char_class.hpp>
#include <boost/spirit/home/karma/detail/generate_to.hpp>

namespace boost { namespace spirit { namespace karma { namespace detail
{
struct pass_through_filter
{
template <typename Char>
Char operator()(Char ch) const
{
return ch;
}
};

template <typename CharEncoding, typename Tag>
struct encoding_filter
{
template <typename Char>
Char operator()(Char ch) const
{
return spirit::char_class::convert<CharEncoding>::to(Tag(), ch);
}
};

template <typename OutputIterator, typename Char, typename Filter>
inline bool string_generate(OutputIterator& sink, Char const* str
, Filter filter)
{
for (Char ch = *str; ch != 0; ch = *++str)
{
*sink = filter(ch);
++sink;
}
return detail::sink_is_good(sink);
}

template <typename OutputIterator, typename Container, typename Filter>
inline bool string_generate(OutputIterator& sink
, Container const& c, Filter filter)
{
typedef typename traits::container_iterator<Container const>::type 
iterator;

const iterator end = boost::end(c);
for (iterator it = boost::begin(c); it != end; ++it)
{
*sink = filter(*it);
++sink;
}
return detail::sink_is_good(sink);
}

template <typename OutputIterator, typename Char>
inline bool string_generate(OutputIterator& sink, Char const* str)
{
return string_generate(sink, str, pass_through_filter());
}

template <typename OutputIterator, typename Container>
inline bool string_generate(OutputIterator& sink
, Container const& c)
{
return string_generate(sink, c, pass_through_filter());
}

template <typename OutputIterator, typename Char, typename CharEncoding
, typename Tag>
inline bool string_generate(OutputIterator& sink
, Char const* str
, CharEncoding, Tag)
{
return string_generate(sink, str, encoding_filter<CharEncoding, Tag>());
}

template <typename OutputIterator, typename Container
, typename CharEncoding, typename Tag>
inline bool 
string_generate(OutputIterator& sink
, Container const& c
, CharEncoding, Tag)
{
return string_generate(sink, c, encoding_filter<CharEncoding, Tag>());
}

template <typename OutputIterator, typename Char>
inline bool string_generate(OutputIterator& sink
, Char const* str
, unused_type, unused_type)
{
return string_generate(sink, str);
}

template <typename OutputIterator, typename Container>
inline bool string_generate(OutputIterator& sink
, Container const& c
, unused_type, unused_type)
{
return string_generate(sink, c);
}

}}}}

#endif
