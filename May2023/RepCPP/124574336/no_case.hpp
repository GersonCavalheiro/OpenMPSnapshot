
#ifndef BOOST_SPIRIT_QI_DIRECTIVE_NO_CASE_HPP
#define BOOST_SPIRIT_QI_DIRECTIVE_NO_CASE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>

namespace boost { namespace spirit
{
template <typename CharEncoding>
struct use_directive<
qi::domain, tag::char_code<tag::no_case, CharEncoding> > 
: mpl::true_ {};

template <typename CharEncoding>
struct is_modifier_directive<qi::domain, tag::char_code<tag::no_case, CharEncoding> >
: mpl::true_ {};
}}

#endif
