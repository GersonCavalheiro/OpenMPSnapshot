
#ifndef BOOST_SPIRIT_QI_OPERATOR_PERMUTATION_HPP
#define BOOST_SPIRIT_QI_OPERATOR_PERMUTATION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/detail/permute_function.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/algorithm/any_if_ns.hpp>
#include <boost/spirit/home/support/detail/what_function.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/support/info.hpp>
#include <boost/fusion/include/size.hpp>
#include <boost/optional.hpp>
#include <boost/array.hpp>
#include <boost/proto/operators.hpp>
#include <boost/proto/tags.hpp>

namespace boost { namespace spirit
{
template <>
struct use_operator<qi::domain, proto::tag::bitwise_xor> 
: mpl::true_ {};

template <>
struct flatten_tree<qi::domain, proto::tag::bitwise_xor> 
: mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
template <typename Elements>
struct permutation : nary_parser<permutation<Elements> >
{
template <typename Context, typename Iterator>
struct attribute
{
typedef typename traits::build_attribute_sequence<
Elements, Context, traits::permutation_attribute_transform
, Iterator, qi::domain
>::type all_attributes;

typedef typename
traits::build_fusion_vector<all_attributes>::type
type;
};

permutation(Elements const& elements_)
: elements(elements_) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper
, Attribute& attr_) const
{
typedef traits::attribute_not_unused<Context, Iterator> predicate;
detail::permute_function<Iterator, Context, Skipper>
f(first, last, context, skipper);

boost::array<bool, fusion::result_of::size<Elements>::value> flags;
flags.fill(false);

typename traits::wrap_if_not_tuple<Attribute>::type attr_local(attr_);


bool result = false;
f.taken = flags.begin();
while (spirit::any_if_ns(elements, attr_local, f, predicate()))
{
f.taken = flags.begin();
result = true;
}
return result;
}

template <typename Context>
info what(Context& context) const
{
info result("permutation");
fusion::for_each(elements,
spirit::detail::what_function<Context>(result, context));
return result;
}

Elements elements;
};

template <typename Elements, typename Modifiers>
struct make_composite<proto::tag::bitwise_xor, Elements, Modifiers>
: make_nary_composite<Elements, permutation>
{};
}}}

namespace boost { namespace spirit { namespace traits
{
template <typename Elements, typename Attribute>
struct pass_attribute<qi::permutation<Elements>, Attribute>
: wrap_if_not_tuple<Attribute> {};

template <typename Elements>
struct has_semantic_action<qi::permutation<Elements> >
: nary_has_semantic_action<Elements> {};

template <typename Elements, typename Attribute, typename Context
, typename Iterator>
struct handles_container<qi::permutation<Elements>, Attribute, Context
, Iterator>
: nary_handles_container<Elements, Attribute, Context, Iterator> {};
}}}

#endif
