
#if !defined(BOOST_SPIRIT_LEX_PLAIN_TOKENID_MASK_JUN_03_2011_0929PM)
#define BOOST_SPIRIT_LEX_PLAIN_TOKENID_MASK_JUN_03_2011_0929PM

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
#include <boost/range/iterator_range_core.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_enum.hpp>
#include <iterator> 
#include <sstream>

namespace boost { namespace spirit
{

template <typename A0>
struct use_terminal<qi::domain
, terminal_ex<tag::tokenid_mask, fusion::vector1<A0> >
> : mpl::or_<is_integral<A0>, is_enum<A0> > {};

template <>
struct use_lazy_terminal<
qi::domain, tag::tokenid_mask, 1
> : mpl::true_ {};
}}

namespace boost { namespace spirit { namespace qi
{
#ifndef BOOST_SPIRIT_NO_PREDEFINED_TERMINALS
using spirit::tokenid_mask;
#endif
using spirit::tokenid_mask_type;

template <typename Mask>
struct plain_tokenid_mask
: primitive_parser<plain_tokenid_mask<Mask> >
{
template <typename Context, typename Iterator>
struct attribute
{
typedef Mask type;
};

plain_tokenid_mask(Mask const& mask)
: mask(mask) {}

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
if ((t.id() & mask) == id_type(mask))
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
ss << "tokenid_mask(" << mask << ")";
return info("tokenid_mask", ss.str());
}

Mask mask;
};

template <typename Modifiers, typename Mask>
struct make_primitive<terminal_ex<tag::tokenid_mask, fusion::vector1<Mask> >
, Modifiers>
{
typedef plain_tokenid_mask<Mask> result_type;

template <typename Terminal>
result_type operator()(Terminal const& term, unused_type) const
{
return result_type(fusion::at_c<0>(term.args));
}
};
}}}

namespace boost { namespace spirit { namespace traits
{
template<typename Mask, typename Attr, typename Context, typename Iterator>
struct handles_container<qi::plain_tokenid_mask<Mask>, Attr, Context, Iterator>
: mpl::true_
{};
}}}

#endif
