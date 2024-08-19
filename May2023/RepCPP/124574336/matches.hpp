
#ifndef BOOST_SPIRIT_QI_DIRECTIVE_MATCHES_HPP
#define BOOST_SPIRIT_QI_DIRECTIVE_MATCHES_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>

namespace boost { namespace spirit
{
template <>
struct use_directive<qi::domain, tag::matches> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::matches;
#endif
using spirit::matches_type;

template <typename Subject>
struct matches_directive : unary_parser<matches_directive<Subject> >
{
typedef Subject subject_type;
matches_directive(Subject const& subject_)
: subject(subject_) {}

template <typename Context, typename Iterator>
struct attribute
{
typedef bool type;
};

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper, Attribute& attr_) const
{
bool result = subject.parse(first, last, context, skipper, unused);
spirit::traits::assign_to(result, attr_);
return true;
}

template <typename Context>
info what(Context& context) const
{
return info("matches", subject.what(context));
}

Subject subject;

BOOST_DELETED_FUNCTION(matches_directive& operator= (matches_directive const&))
};

template <typename Subject, typename Modifiers>
struct make_directive<tag::matches, Subject, Modifiers>
{
typedef matches_directive<Subject> result_type;
result_type operator()(unused_type, Subject const& subject, unused_type) const
{
return result_type(subject);
}
};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Subject>
struct has_semantic_action<qi::matches_directive<Subject> >
: unary_has_semantic_action<Subject> {};

template <typename Subject, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::matches_directive<Subject>, Attribute
, Context, Iterator>
: unary_handles_container<Subject, Attribute, Context, Iterator> {};
}}}

#endif
