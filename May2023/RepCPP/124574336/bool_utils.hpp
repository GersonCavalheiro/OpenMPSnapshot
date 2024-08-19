
#if !defined(BOOST_SPIRIT_KARMA_BOOL_UTILS_SEP_28_2009_0644PM)
#define BOOST_SPIRIT_KARMA_BOOL_UTILS_SEP_28_2009_0644PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/char_class.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/karma/detail/generate_to.hpp>
#include <boost/spirit/home/karma/detail/string_generate.hpp>
#include <boost/spirit/home/karma/numeric/detail/numeric_utils.hpp>
#include <boost/detail/workaround.hpp>

namespace boost { namespace spirit { namespace karma 
{ 
template <typename T>
struct bool_policies;

template <typename T
, typename Policies = bool_policies<T>
, typename CharEncoding = unused_type
, typename Tag = unused_type>
struct bool_inserter
{
template <typename OutputIterator, typename U>
static bool
call (OutputIterator& sink, U b, Policies const& p = Policies())
{
#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1600))
p; 
#endif
return p.template call<bool_inserter>(sink, T(b), p);
}

template <typename OutputIterator, typename U>
static bool
call_n (OutputIterator& sink, U b, Policies const& p)
{
#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1600))
p; 
#endif
if (b) 
return p.template generate_true<CharEncoding, Tag>(sink, b);
return p.template generate_false<CharEncoding, Tag>(sink, b);
}
};

}}}

#endif

