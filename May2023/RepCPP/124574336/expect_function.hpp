
#ifndef BOOST_SPIRIT_QI_DETAIL_EXPECT_FUNCTION_HPP
#define BOOST_SPIRIT_QI_DETAIL_EXPECT_FUNCTION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>
#include <boost/spirit/home/support/multi_pass_wrapper.hpp>
#include <boost/throw_exception.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <
typename Iterator, typename Context
, typename Skipper, typename Exception>
struct expect_function
{
typedef Iterator iterator_type;
typedef Context context_type;

expect_function(
Iterator& first_, Iterator const& last_
, Context& context_, Skipper const& skipper_)
: first(first_)
, last(last_)
, context(context_)
, skipper(skipper_)
, is_first(true)
{
}

template <typename Component, typename Attribute>
bool operator()(Component const& component, Attribute& attr) const
{
if (!is_first)
spirit::traits::clear_queue(first);

if (!component.parse(first, last, context, skipper, attr))
{
if (is_first)
{
is_first = false;
return true;        
}
boost::throw_exception(Exception(first, last, component.what(context)));
#if defined(BOOST_NO_EXCEPTIONS)
return true;            
#endif
}
is_first = false;
return false;
}

template <typename Component>
bool operator()(Component const& component) const
{
if (!is_first)
spirit::traits::clear_queue(first);

if (!component.parse(first, last, context, skipper, unused))
{
if (is_first)
{
is_first = false;
return true;
}
boost::throw_exception(Exception(first, last, component.what(context)));
#if defined(BOOST_NO_EXCEPTIONS)
return false;   
#endif
}
is_first = false;
return false;
}

Iterator& first;
Iterator const& last;
Context& context;
Skipper const& skipper;
mutable bool is_first;

BOOST_DELETED_FUNCTION(expect_function& operator= (expect_function const&))
};
}}}}

#endif
