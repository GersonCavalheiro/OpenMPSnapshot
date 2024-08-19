
#if !defined(BOOST_SPIRIT_HANDLES_CONTAINER_DEC_18_2010_0920AM)
#define BOOST_SPIRIT_HANDLES_CONTAINER_DEC_18_2010_0920AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/attributes_fwd.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/not.hpp>
#include <boost/mpl/find_if.hpp>
#include <boost/type_traits/is_same.hpp>

namespace boost { namespace spirit { namespace traits
{
template <typename T, typename Attribute, typename Context
, typename Iterator, typename Enable>
struct handles_container : mpl::false_ {};

template <typename Subject, typename Attribute, typename Context
, typename Iterator>
struct unary_handles_container
: handles_container<Subject, Attribute, Context, Iterator> {};

template <typename Left, typename Right, typename Attribute
, typename Context, typename Iterator>
struct binary_handles_container 
: mpl::or_<
handles_container<Left, Attribute, Context, Iterator>
, handles_container<Right, Attribute, Context, Iterator> > 
{};

template <typename Elements, typename Attribute, typename Context
, typename Iterator>
struct nary_handles_container
: mpl::not_<
is_same<
typename mpl::find_if<
Elements, handles_container<mpl::_, Attribute
, Context, Iterator> 
>::type
, typename mpl::end<Elements>::type> > 
{};
}}}

#endif
