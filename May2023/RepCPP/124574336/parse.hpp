
#if !defined(BOOST_SPIRIT_DETAIL_PARSE_DEC_02_2009_0411PM)
#define BOOST_SPIRIT_DETAIL_PARSE_DEC_02_2009_0411PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/meta_compiler.hpp>
#include <boost/spirit/home/qi/skip_flag.hpp>
#include <boost/spirit/home/qi/skip_over.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/bool.hpp>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Expr, typename Enable = void>
struct parse_impl
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);
};

template <typename Expr>
struct parse_impl<Expr
, typename enable_if<traits::matches<qi::domain, Expr> >::type>
{
template <typename Iterator>
static bool call(
Iterator& first
, Iterator last
, Expr const& expr)
{
return compile<qi::domain>(expr).parse(
first, last, unused, unused, unused);
}
};

template <typename Expr, typename Enable = void>
struct phrase_parse_impl
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);
};

template <typename Expr>
struct phrase_parse_impl<Expr
, typename enable_if<traits::matches<qi::domain, Expr> >::type>
{
template <typename Iterator, typename Skipper>
static bool call(
Iterator& first
, Iterator last
, Expr const& expr
, Skipper const& skipper
, BOOST_SCOPED_ENUM(skip_flag) post_skip)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Skipper);

typedef
typename result_of::compile<qi::domain, Skipper>::type
skipper_type;
skipper_type const skipper_ = compile<qi::domain>(skipper);

if (!compile<qi::domain>(expr).parse(
first, last, unused, skipper_, unused))
return false;

if (post_skip == skip_flag::postskip)
qi::skip_over(first, last, skipper_);
return true;
}
};

}}}}

#endif

