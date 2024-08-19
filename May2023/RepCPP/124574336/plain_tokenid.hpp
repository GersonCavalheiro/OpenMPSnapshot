
#if !defined(BOOST_SPIRIT_LEX_PLAIN_TOKENID_NOV_26_2010_0944AM)
#define BOOST_SPIRIT_LEX_PLAIN_TOKENID_NOV_26_2010_0944AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/info.hpp>
#include <boost/spirit/home/qi/detail/attributes.hpp>
#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/handles_container.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/qi/domain.hpp>
#include <boost/spirit/home/qi/parser.hpp>
#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/detail/assign_to.hpp>

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/and.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <iterator> 
#include <sstream>

namespace boost { namespace spirit
{

template <>
struct use_terminal<qi::domain, tag::tokenid>
: mpl::true_ {};

template <typename A0>
struct use_terminal<qi::domain
, terminal_ex<tag::tokenid, fusion::vector1<A0> >
> : mpl::or_<is_integral<A0>, is_enum<A0> > {};

template <typename A0, typename A1>
struct use_terminal<qi::domain
, terminal_ex<tag::tokenid, fusion::vector2<A0, A1> >
> : mpl::and_<
mpl::or_<is_integral<A0>, is_enum<A0> >
, mpl::or_<is_integral<A1>, is_enum<A1> >
> {};

template <>
struct use_lazy_terminal<
qi::domain, tag::tokenid, 1
> : mpl::true_ {};

template <>
struct use_lazy_terminal<
qi::domain, tag::tokenid, 2
> : mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::tokenid;
#endif
using spirit::tokenid_type;

template <typename TokenId>
struct plain_tokenid
: primitive_parser<plain_tokenid<TokenId> >
{
template <typename Context, typename Iterator>
struct attribute
{
typedef TokenId type;
};

plain_tokenid(TokenId const& id)
: id(id) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& , Skipper const& skipper
, Attribute& attr) const
{
qi::skip_over(first, last, skipper);   

if (first != last) {

typedef typename
std::iterator_traits<Iterator>::value_type
token_type;
typedef typename token_type::id_type id_type;

token_type const& t = *first;
if (id_type(~0) == id_type(id) || id_type(id) == t.id()) {
spirit::traits::assign_to(id, attr);
++first;
return true;
}
}
return false;
}

template <typename Context>
info what(Context& ) const
{
std::stringstream ss;
ss << "tokenid(" << id << ")";
return info("tokenid", ss.str());
}

TokenId id;
};

template <typename TokenId>
struct plain_tokenid_range
: primitive_parser<plain_tokenid_range<TokenId> >
{
template <typename Context, typename Iterator>
struct attribute
{
typedef TokenId type;
};

plain_tokenid_range(TokenId const& idmin, TokenId const& idmax)
: idmin(idmin), idmax(idmax) {}

template <typename Iterator, typename Context
, typename Skipper, typename Attribute>
bool parse(Iterator& first, Iterator const& last
, Context& , Skipper const& skipper
, Attribute& attr) const
{
qi::skip_over(first, last, skipper);   

if (first != last) {

typedef typename
std::iterator_traits<Iterator>::value_type
token_type;
typedef typename token_type::id_type id_type;

token_type const& t = *first;
if (id_type(idmin) >= t.id() && id_type(idmin) <= t.id())
{
spirit::traits::assign_to(t.id(), attr);
++first;
return true;
}
}
return false;
}

template <typename Context>
info what(Context& ) const
{
std::stringstream ss;
ss << "token(" << idmin << ", " << idmax << ")";
return info("tokenid_range", ss.str());
}

TokenId idmin, idmax;
};

template <typename Modifiers>
struct make_primitive<tag::tokenid, Modifiers>
{
typedef plain_tokenid<std::size_t> result_type;

result_type operator()(unused_type, unused_type) const
{
return result_type(std::size_t(~0));
}
};

template <typename Modifiers, typename TokenId>
struct make_primitive<terminal_ex<tag::tokenid, fusion::vector1<TokenId> >
, Modifiers>
{
typedef plain_tokenid<TokenId> result_type;

template <typename Terminal>
result_type operator()(Terminal const& term, unused_type) const
{
return result_type(fusion::at_c<0>(term.args));
}
};

template <typename Modifiers, typename TokenId>
struct make_primitive<terminal_ex<tag::tokenid, fusion::vector2<TokenId, TokenId> >
, Modifiers>
{
typedef plain_tokenid_range<TokenId> result_type;

template <typename Terminal>
result_type operator()(Terminal const& term, unused_type) const
{
return result_type(fusion::at_c<0>(term.args)
, fusion::at_c<1>(term.args));
}
};
}}}

namespace boost { namespace spirit { namespace traits
{
template<typename Idtype, typename Attr, typename Context, typename Iterator>
struct handles_container<qi::plain_tokenid<Idtype>, Attr, Context, Iterator>
: mpl::true_
{};

template<typename Idtype, typename Attr, typename Context, typename Iterator>
struct handles_container<qi::plain_tokenid_range<Idtype>, Attr, Context, Iterator>
: mpl::true_
{};
}}}

#endif
