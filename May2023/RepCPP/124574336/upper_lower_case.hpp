
#ifndef BOOST_SPIRIT_KARMA_DIRECTIVE_UPPER_LOWER_CASE_HPP
#define BOOST_SPIRIT_KARMA_DIRECTIVE_UPPER_LOWER_CASE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/modify.hpp>
#include <boost/spirit/home/karma/domain.hpp>
#include <boost/spirit/home/karma/meta_compiler.hpp>

namespace boost { namespace spirit
{
template <typename CharEncoding>
struct use_directive<
karma::domain, tag::char_code<tag::upper, CharEncoding> > 
: mpl::true_ {};

template <typename CharEncoding>
struct use_directive<
karma::domain, tag::char_code<tag::lower, CharEncoding> > 
: mpl::true_ {};

template <typename CharEncoding>
struct is_modifier_directive<karma::domain
, tag::char_code<tag::upper, CharEncoding> >
: mpl::true_ {};

template <typename CharEncoding>
struct is_modifier_directive<karma::domain
, tag::char_code<tag::lower, CharEncoding> >
: mpl::true_ {};

template <typename Current, typename CharEncoding>
struct compound_modifier<
Current
, tag::char_code<tag::upper, CharEncoding>
, typename enable_if<
has_modifier<Current, tag::char_code<tag::lower, CharEncoding> > 
>::type
>
: Current
{
compound_modifier()
: Current() {}

compound_modifier(Current const& current, 
tag::char_code<tag::upper, CharEncoding> const&)
: Current(current) {}
};

template <typename Current, typename CharEncoding>
struct compound_modifier<
Current
, tag::char_code<tag::lower, CharEncoding>
, typename enable_if<
has_modifier<Current, tag::char_code<tag::upper, CharEncoding> > 
>::type
>
: Current
{
compound_modifier()
: Current() {}

compound_modifier(Current const& current, 
tag::char_code<tag::lower, CharEncoding> const&)
: Current(current) {}
};
}}

#endif
