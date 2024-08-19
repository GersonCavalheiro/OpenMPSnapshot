
#ifndef BOOST_SPIRIT_QI_OPERATOR_OPTIONAL_HPP
#define BOOST_SPIRIT_QI_OPERATOR_OPTIONAL_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/support/container.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>
#include <boost/optional.hpp>
#include <boost/proto/operators.hpp>
#include <boost/proto/tags.hpp>
#include <vector>

namespace boost { namespace spirit
{
template <>
struct use_operator<qi::domain, proto::tag::negate> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
template <typename Subject>
struct optional : unary_parser<optional<Subject> >
{
typedef Subject subject_type;

template <typename Context, typename Iterator>
struct attribute
{
typedef typename
traits::build_optional<
typename traits::
attribute_of<Subject, Context, Iterator>::type
>::type
type;
};

optional(Subject const& subject_)
: subject(subject_) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse_impl(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_, mpl::false_) const
{
typename spirit::result_of::optional_value<Attribute>::type val =
typename spirit::result_of::optional_value<Attribute>::type();

if (subject.parse(first, last, context, skipper, val))
{
spirit::traits::assign_to(val, attr_);
}
return true;
}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse_impl(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_, mpl::true_) const
{
subject.parse(first, last, context, skipper, attr_);
return true;
}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_) const
{
typedef typename spirit::result_of::optional_value<Attribute>::type
attribute_type;

return parse_impl(first, last, context, skipper, attr_
, traits::is_container<attribute_type>());
}

template <typename Context>
info what(Context& context) const
{
return info("optional", subject.what(context));
}

Subject subject;
};

template <typename Elements, typename Modifiers>
struct make_composite<proto::tag::negate, Elements, Modifiers>
: make_unary_composite<Elements, optional>
{};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Subject>
struct has_semantic_action<qi::optional<Subject> >
: unary_has_semantic_action<Subject> {};

template <typename Subject, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::optional<Subject>, Attribute
, Context, Iterator>
: mpl::true_ {};
}}}

#endif
