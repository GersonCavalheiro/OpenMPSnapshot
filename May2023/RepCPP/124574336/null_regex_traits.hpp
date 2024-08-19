
#ifndef BOOST_XPRESSIVE_TRAITS_NULL_REGEX_TRAITS_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_TRAITS_NULL_REGEX_TRAITS_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
#endif

#include <vector>
#include <boost/assert.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/utility/never_true.hpp>
#include <boost/xpressive/detail/utility/ignore_unused.hpp>

namespace boost { namespace xpressive
{

namespace detail
{
struct not_a_locale {};
}

struct regex_traits_version_1_tag;

template<typename Elem>
struct null_regex_traits
{
typedef Elem char_type;
typedef std::vector<char_type> string_type;
typedef detail::not_a_locale locale_type;
typedef int char_class_type;
typedef regex_traits_version_1_tag version_tag;

null_regex_traits(locale_type = locale_type())
{
}

bool operator ==(null_regex_traits<char_type> const &that) const
{
detail::ignore_unused(that);
return true;
}

bool operator !=(null_regex_traits<char_type> const &that) const
{
detail::ignore_unused(that);
return false;
}

char_type widen(char ch) const
{
return char_type(ch);
}

static unsigned char hash(char_type ch)
{
return static_cast<unsigned char>(ch);
}

static char_type translate(char_type ch)
{
return ch;
}

static char_type translate_nocase(char_type ch)
{
return ch;
}

static bool in_range(char_type first, char_type last, char_type ch)
{
return first <= ch && ch <= last;
}

static bool in_range_nocase(char_type first, char_type last, char_type ch)
{
return first <= ch && ch <= last;
}

template<typename FwdIter>
static string_type transform(FwdIter begin, FwdIter end)
{
return string_type(begin, end);
}

template<typename FwdIter>
static string_type transform_primary(FwdIter begin, FwdIter end)
{
return string_type(begin, end);
}

template<typename FwdIter>
static string_type lookup_collatename(FwdIter begin, FwdIter end)
{
detail::ignore_unused(begin);
detail::ignore_unused(end);
return string_type();
}

template<typename FwdIter>
static char_class_type lookup_classname(FwdIter begin, FwdIter end, bool icase)
{
detail::ignore_unused(begin);
detail::ignore_unused(end);
detail::ignore_unused(icase);
return 0;
}

static bool isctype(char_type ch, char_class_type mask)
{
detail::ignore_unused(ch);
detail::ignore_unused(mask);
return false;
}

static int value(char_type ch, int radix)
{
detail::ignore_unused(ch);
detail::ignore_unused(radix);
return -1;
}

static locale_type imbue(locale_type loc)
{
return loc;
}

static locale_type getloc()
{
return locale_type();
}
};

}}

#endif
