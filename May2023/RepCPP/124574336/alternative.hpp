
#ifndef BOOST_SPIRIT_QI_OPERATOR_ALTERNATIVE_HPP
#define BOOST_SPIRIT_QI_OPERATOR_ALTERNATIVE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/detail/alternative_function.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/support/detail/what_function.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/fusion/include/any.hpp>
#include <boost/fusion/include/mpl.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/proto/operators.hpp>
#include <boost/proto/tags.hpp>

namespace boost { namespace spirit
{
template <>
struct use_operator<qi::domain, proto::tag::bitwise_or> 
: mpl::true_ {};

template <>
struct flatten_tree<qi::domain, proto::tag::bitwise_or> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
template <typename Elements>
struct alternative : nary_parser<alternative<Elements> >
{
template <typename Context, typename Iterator>
struct attribute
{
typedef typename traits::build_attribute_sequence<
Elements, Context, traits::alternative_attribute_transform
, Iterator, qi::domain
>::type all_attributes;

typedef typename
traits::build_variant<all_attributes>::type
type;
};

alternative(Elements const& elements_)
: elements(elements_) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_) const
{
detail::alternative_function<Iterator, Context, Skipper, Attribute>
f(first, last, context, skipper, attr_);

return fusion::any(elements, f);
}

template <typename Context>
info what(Context& context) const
{
info result("alternative");
fusion::for_each(elements,
spirit::detail::what_function<Context>(result, context));
return result;
}

Elements elements;
};

template <typename Elements, typename Modifiers>
struct make_composite<proto::tag::bitwise_or, Elements, Modifiers>
: make_nary_composite<Elements, alternative>
{};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Elements>
struct has_semantic_action<qi::alternative<Elements> >
: nary_has_semantic_action<Elements> {};

template <typename Elements, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::alternative<Elements>, Attribute, Context
, Iterator>
: nary_handles_container<Elements, Attribute, Context, Iterator> {};
}}}

#endif
