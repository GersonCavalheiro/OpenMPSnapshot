
#if !defined(BOOST_SPIRIT_PARSER_BINDER_DECEMBER_05_2008_0516_PM)
#define BOOST_SPIRIT_PARSER_BINDER_DECEMBER_05_2008_0516_PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/fusion/include/at.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/spirit/home/support/has_semantic_action.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Parser, typename Auto>
struct parser_binder
{
parser_binder(Parser const& p_)
: p(p_) {}

template <typename Iterator, typename Skipper, typename Context>
bool call(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper, mpl::true_) const
{
return p.parse(first, last, context, skipper, unused);
}

template <typename Iterator, typename Skipper, typename Context>
bool call(Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper, mpl::false_) const
{
return p.parse(first, last, context, skipper
, fusion::at_c<0>(context.attributes));
}

template <typename Iterator, typename Skipper, typename Context>
bool operator()(
Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper) const
{
typedef typename traits::has_semantic_action<Parser>::type auto_rule;
return call(first, last, context, skipper, auto_rule());
}

Parser p;
};

template <typename Parser>
struct parser_binder<Parser, mpl::true_>
{
parser_binder(Parser const& p_)
: p(p_) {}

template <typename Iterator, typename Skipper, typename Context>
bool operator()(
Iterator& first, Iterator const& last
, Context& context, Skipper const& skipper) const
{
return p.parse(first, last, context, skipper
, fusion::at_c<0>(context.attributes));
}

Parser p;
};

template <typename Auto, typename Parser>
inline parser_binder<Parser, Auto>
bind_parser(Parser const& p)
{
return parser_binder<Parser, Auto>(p);
}
}}}}

#endif
