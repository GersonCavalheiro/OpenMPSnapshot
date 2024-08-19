
#ifndef BOOST_SPIRIT_SUPPORT_DETAIL_WHAT_FUNCTION_HPP
#define BOOST_SPIRIT_SUPPORT_DETAIL_WHAT_FUNCTION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <string>
#include <boost/spirit/home/support/info.hpp>
#include <boost/detail/workaround.hpp>

namespace boost { namespace spirit { namespace detail
{
template <typename Context>
struct what_function
{
what_function(info& what_, Context& context_)
: what(what_), context(context_)
{
what.value = std::list<info>();
}

template <typename Component>
void operator()(Component const& component) const
{
#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1600))
component; 
#endif
boost::get<std::list<info> >(what.value).
push_back(component.what(context));
}

info& what;
Context& context;

BOOST_DELETED_FUNCTION(what_function& operator= (what_function const&))
};
}}}

#endif
