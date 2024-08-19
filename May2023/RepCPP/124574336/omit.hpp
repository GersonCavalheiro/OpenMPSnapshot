
#ifndef BOOST_SPIRIT_QI_DIRECTIVE_OMIT_HPP
#define BOOST_SPIRIT_QI_DIRECTIVE_OMIT_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>

namespace boost { namespace spirit
{
template <>
struct use_directive<qi::domain, tag::omit> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::omit;
#endif
using spirit::omit_type;

template <typename Subject>
struct omit_directive : unary_parser<omit_directive<Subject> >
{
typedef Subject subject_type;
omit_directive(Subject const& subject_)
: subject(subject_) {}

template <typename Context, typename Iterator>
struct attribute
{
typedef unused_type type;
};

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper, Attribute& attr_) const
{
return subject.parse(first, last, context, skipper, attr_);
}

template <typename Context>
info what(Context& context) const
{
return info("omit", subject.what(context));
}

Subject subject;

BOOST_DELETED_FUNCTION(omit_directive& operator= (omit_directive const&))
};

template <typename Subject, typename Modifiers>
struct make_directive<tag::omit, Subject, Modifiers>
{
typedef omit_directive<Subject> result_type;
result_type operator()(unused_type, Subject const& subject, unused_type) const
{
return result_type(subject);
}
};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Subject>
struct has_semantic_action<qi::omit_directive<Subject> >
: mpl::false_ {};

template <typename Subject, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::omit_directive<Subject>, Attribute
, Context, Iterator>
: unary_handles_container<Subject, Attribute, Context, Iterator> {};
}}}

#endif
