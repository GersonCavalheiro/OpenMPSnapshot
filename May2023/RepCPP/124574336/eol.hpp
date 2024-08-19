
#if !defined(BOOST_SPIRIT_EOL_APRIL_18_2008_0751PM)
#define BOOST_SPIRIT_EOL_APRIL_18_2008_0751PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/mpl/bool.hpp>
#include <boost/spirit/home/qi/domain.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>

namespace boost { namespace spirit
{
template <>
struct use_terminal<qi::domain, tag::eol>       
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::eol;
#endif
using spirit::eol_type;

struct eol_parser : primitive_parser<eol_parser>
{
template <typename Context, typename Iterator>
struct attribute
{
typedef unused_type type;
};

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& , Skipper const& skipper
, Attribute& ) const
{
qi::skip_over(first, last, skipper);

Iterator it = first;
bool matched = false;
if (it != last && *it == '\r')  
{
matched = true;
++it;
}
if (it != last && *it == '\n')  
{
matched = true;
++it;
}

if (!matched)
return false;

first = it;
return true;
}

template <typename Context>
info what(Context& ) const
{
return info("eol");
}
};

template <typename Modifiers>
struct make_primitive<tag::eol, Modifiers>
{
typedef eol_parser result_type;
result_type operator()(unused_type, unused_type) const
{
return result_type();
}
};
}}}

#endif


