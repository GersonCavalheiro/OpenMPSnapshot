
#ifndef BOOST_SPIRIT_QI_DETAIL_PERMUTE_FUNCTION_HPP
#define BOOST_SPIRIT_QI_DETAIL_PERMUTE_FUNCTION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/optional.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Iterator, typename Context, typename Skipper>
struct permute_function
{
permute_function(
Iterator& first_, Iterator const& last_
, Context& context_, Skipper const& skipper_)
: first(first_)
, last(last_)
, context(context_)
, skipper(skipper_)
{
}

template <typename Component, typename Attribute>
bool operator()(Component const& component, Attribute& attr)
{
if (!*taken && component.parse(first, last, context, skipper, attr))
{
*taken = true;
++taken;
return true;
}
++taken;
return false;
}

template <typename Component>
bool operator()(Component const& component)
{
if (!*taken && component.parse(first, last, context, skipper, unused))
{
*taken = true;
++taken;
return true;
}
++taken;
return false;
}

Iterator& first;
Iterator const& last;
Context& context;
Skipper const& skipper;
bool* taken;

BOOST_DELETED_FUNCTION(permute_function& operator= (permute_function const&))
};
}}}}

#endif
