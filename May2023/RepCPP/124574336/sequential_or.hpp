
#ifndef BOOST_SPIRIT_QI_OPERATOR_SEQUENTIAL_OR_HPP
#define BOOST_SPIRIT_QI_OPERATOR_SEQUENTIAL_OR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/detail/pass_function.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/detail/what_function.hpp>
#include <boost/spirit/home/support/algorithm/any_if_ns_so.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/fusion/include/as_vector.hpp>
#include <boost/fusion/include/for_each.hpp>
#include <boost/proto/operators.hpp>
#include <boost/proto/tags.hpp>

namespace boost { namespace spirit
{
template <>
struct use_operator<qi::domain, proto::tag::logical_or> 
: mpl::true_ {};

template <>
struct flatten_tree<qi::domain, proto::tag::logical_or> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
template <typename Elements>
struct sequential_or : nary_parser<sequential_or<Elements> >
{
template <typename Context, typename Iterator>
struct attribute
{
typedef typename traits::build_attribute_sequence<
Elements, Context, traits::sequential_or_attribute_transform
, Iterator, qi::domain
>::type all_attributes;

typedef typename
traits::build_fusion_vector<all_attributes>::type
type;
};

sequential_or(Elements const& elements_)
: elements(elements_) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_) const
{
typedef traits::attribute_not_unused<Context, Iterator> predicate;
detail::pass_function<Iterator, Context, Skipper>
f(first, last, context, skipper);

typename traits::wrap_if_not_tuple<Attribute>::type attr_local(attr_);

return spirit::any_if_ns_so(elements, attr_local, f, predicate());
}

template <typename Context>
info what(Context& context) const
{
info result("sequential-or");
fusion::for_each(elements,
spirit::detail::what_function<Context>(result, context));
return result;
}

Elements elements;
};

template <typename Elements, typename Modifiers>
struct make_composite<proto::tag::logical_or, Elements, Modifiers>
: make_nary_composite<Elements, sequential_or>
{};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Elements, typename Attribute>
struct pass_attribute<qi::sequential_or<Elements>, Attribute>
: wrap_if_not_tuple<Attribute> {};

template <typename Elements>
struct has_semantic_action<qi::sequential_or<Elements> >
: nary_has_semantic_action<Elements> {};

template <typename Elements, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::sequential_or<Elements>, Attribute, Context
, Iterator>
: nary_handles_container<Elements, Attribute, Context, Iterator> {};
}}}

#endif
