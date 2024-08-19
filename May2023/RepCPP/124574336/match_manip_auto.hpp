
#if !defined(BOOST_SPIRIT_MATCH_MANIP_AUTO_DEC_02_2009_0813PM)
#define BOOST_SPIRIT_MATCH_MANIP_AUTO_DEC_02_2009_0813PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/stream/detail/match_manip.hpp>
#include <boost/spirit/home/qi/auto/create_parser.hpp>
#include <boost/utility/enable_if.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Expr>
struct match<Expr
, typename enable_if<traits::meta_create_exists<qi::domain, Expr> >::type>
{
typedef typename result_of::create_parser<Expr>::type expr_type;
typedef match_manip<
expr_type, mpl::true_, mpl::false_, unused_type, Expr
> type;

static type call(Expr const& expr)
{
return type(create_parser<Expr>(), unused, const_cast<Expr&>(expr));
}
};

template <typename Expr, typename Skipper>
struct phrase_match<Expr, Skipper
, typename enable_if<traits::meta_create_exists<qi::domain, Expr> >::type>
{
typedef typename result_of::create_parser<Expr>::type expr_type;
typedef match_manip<
expr_type, mpl::true_, mpl::false_, Skipper, Expr
> type;

static type call(
Expr const& expr
, Skipper const& skipper
, BOOST_SCOPED_ENUM(skip_flag) post_skip)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Skipper);
return type(create_parser<Expr>(), skipper, post_skip
, const_cast<Expr&>(expr));
}
};

}}}}

#endif
