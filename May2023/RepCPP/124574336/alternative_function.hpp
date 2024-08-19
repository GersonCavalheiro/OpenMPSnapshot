
#ifndef BOOST_SPIRIT_QI_DETAIL_ALTERNATIVE_FUNCTION_HPP
#define BOOST_SPIRIT_QI_DETAIL_ALTERNATIVE_FUNCTION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/domain.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/bool.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Variant, typename Expected>
struct find_substitute
{

typedef Variant variant_type;
typedef typename variant_type::types types;
typedef typename mpl::end<types>::type end;

typedef typename mpl::find<types, Expected>::type iter_1;

typedef typename
mpl::eval_if<
is_same<iter_1, end>,
mpl::find_if<types, traits::is_substitute<mpl::_1, Expected> >,
mpl::identity<iter_1>
>::type
iter;

typedef typename
mpl::eval_if<
is_same<iter, end>,
mpl::identity<Expected>,
mpl::deref<iter>
>::type
type;
};

template <typename Iterator, typename Context, typename Skipper,
typename Attribute>
struct alternative_function
{
alternative_function(
Iterator& first_, Iterator const& last_, Context& context_,
Skipper const& skipper_, Attribute& attr_)
: first(first_), last(last_), context(context_), skipper(skipper_),
attr(attr_)
{
}

template <typename Component>
bool call(Component const& component, mpl::true_) const
{
return component.parse(first, last, context, skipper, attr);
}

template <typename Component>
bool call_optional_or_variant(Component const& component, mpl::true_) const
{
typedef typename
traits::attribute_of<Component, Context, Iterator>::type
expected_type;

typename mpl::if_<
is_same<expected_type, unused_type>,
unused_type,
typename Attribute::value_type>::type
val;

if (component.parse(first, last, context, skipper, val))
{
traits::assign_to(val, attr);
return true;
}
return false;
}

template <typename Component>
bool call_variant(Component const& component, mpl::false_) const
{

typename
find_substitute<Attribute,
typename traits::attribute_of<Component, Context, Iterator>::type
>::type
val;

if (component.parse(first, last, context, skipper, val))
{
traits::assign_to(val, attr);
return true;
}
return false;
}

template <typename Component>
bool call_variant(Component const& component, mpl::true_) const
{

return component.parse(first, last, context, skipper, attr);
}

template <typename Component>
bool call_optional_or_variant(Component const& component, mpl::false_) const
{

typedef typename
traits::attribute_of<Component, Context, Iterator>::type
expected;
return call_variant(component,
is_same<Attribute, expected>());
}

template <typename Component>
bool call(Component const& component, mpl::false_) const
{
return call_optional_or_variant(
component, spirit::traits::not_is_variant<Attribute, qi::domain>());
}

template <typename Component>
bool call_unused(Component const& component, mpl::true_) const
{
return call(component,
mpl::and_<
spirit::traits::not_is_variant<Attribute, qi::domain>,
spirit::traits::not_is_optional<Attribute, qi::domain>
>());
}

template <typename Component>
bool call_unused(Component const& component, mpl::false_) const
{
return component.parse(first, last, context, skipper, unused);
}

template <typename Component>
bool operator()(Component const& component) const
{
typedef typename traits::not_is_unused<
typename traits::attribute_of<Component, Context, Iterator>::type
>::type predicate;

return call_unused(component, predicate());
}

Iterator& first;
Iterator const& last;
Context& context;
Skipper const& skipper;
Attribute& attr;

BOOST_DELETED_FUNCTION(alternative_function& operator= (alternative_function const&))
};

template <typename Iterator, typename Context, typename Skipper>
struct alternative_function<Iterator, Context, Skipper, unused_type const>
{
alternative_function(
Iterator& first_, Iterator const& last_, Context& context_,
Skipper const& skipper_, unused_type)
: first(first_), last(last_), context(context_), skipper(skipper_)
{
}

template <typename Component>
bool operator()(Component const& component) const
{
return component.parse(first, last, context, skipper,
unused);
}

Iterator& first;
Iterator const& last;
Context& context;
Skipper const& skipper;

BOOST_DELETED_FUNCTION(alternative_function& operator= (alternative_function const&))
};

}}}}

#endif
