
#ifndef BOOST_SPIRIT_QI_NUMERIC_BOOL_POLICIES_HPP
#define BOOST_SPIRIT_QI_NUMERIC_BOOL_POLICIES_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/detail/string_parse.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>

namespace boost { namespace spirit { namespace qi
{
template <typename T = bool>
struct bool_policies
{
template <typename Iterator, typename Attribute>
static bool
parse_true(Iterator& first, Iterator const& last, Attribute& attr_)
{
if (detail::string_parse("true", first, last, unused))
{
spirit::traits::assign_to(T(true), attr_);    
return true;
}
return false;
}

template <typename Iterator, typename Attribute>
static bool
parse_false(Iterator& first, Iterator const& last, Attribute& attr_)
{
if (detail::string_parse("false", first, last, unused))
{
spirit::traits::assign_to(T(false), attr_);   
return true;
}
return false;
}
};

template <typename T = bool>
struct no_case_bool_policies
{
template <typename Iterator, typename Attribute>
static bool
parse_true(Iterator& first, Iterator const& last, Attribute& attr_)
{
if (detail::string_parse("true", "TRUE", first, last, unused))
{
spirit::traits::assign_to(T(true), attr_);    
return true;
}
return false;
}

template <typename Iterator, typename Attribute>
static bool
parse_false(Iterator& first, Iterator const& last, Attribute& attr_)
{
if (detail::string_parse("false", "FALSE", first, last, unused))
{
spirit::traits::assign_to(T(false), attr_);   
return true;
}
return false;
}
};

}}}

#endif
