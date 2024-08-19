
#if !defined(BOOST_SPIRIT_MATCH_MANIP_MAY_05_2007_1203PM)
#define BOOST_SPIRIT_MATCH_MANIP_MAY_05_2007_1203PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/qi/parse.hpp>
#include <boost/spirit/home/support/iterators/istream_iterator.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/mpl/bool.hpp>

#include <iterator>
#include <string>

namespace boost { namespace spirit { namespace qi { namespace detail
{
template <typename Expr
, typename CopyExpr = mpl::false_, typename CopyAttr = mpl::false_
, typename Skipper = unused_type, typename Attribute = unused_type const>
struct match_manip
{
BOOST_SPIRIT_ASSERT_MSG(!CopyExpr::value || !CopyAttr::value
, error_invalid_should_not_happen, ());

match_manip(Expr const& xpr, Skipper const& s, Attribute& a)
: expr(xpr), skipper(s), attr(a), post_skip(skip_flag::postskip) {}

match_manip(Expr const& xpr, Skipper const& s
, BOOST_SCOPED_ENUM(skip_flag) ps, Attribute& a)
: expr(xpr), skipper(s), attr(a), post_skip(ps) {}

Expr const& expr;
Skipper const& skipper;
Attribute& attr;
BOOST_SCOPED_ENUM(skip_flag) const post_skip;

BOOST_DELETED_FUNCTION(match_manip& operator= (match_manip const&))
};

template <typename Expr, typename Skipper, typename Attribute>
struct match_manip<Expr, mpl::false_, mpl::true_, Skipper, Attribute>
{
match_manip(Expr const& xpr, Skipper const& s, Attribute& a)
: expr(xpr), skipper(s), attr(a), post_skip(skip_flag::postskip) {}

match_manip(Expr const& xpr, Skipper const& s
, BOOST_SCOPED_ENUM(skip_flag) ps, Attribute& a)
: expr(xpr), skipper(s), attr(a), post_skip(ps) {}

Expr const& expr;
Skipper const& skipper;
Attribute attr;
BOOST_SCOPED_ENUM(skip_flag) const post_skip;

BOOST_DELETED_FUNCTION(match_manip& operator= (match_manip const&))
};

template <typename Expr, typename Skipper, typename Attribute>
struct match_manip<Expr, mpl::true_, mpl::false_, Skipper, Attribute>
{
match_manip(Expr const& xpr, Skipper const& s, Attribute& a)
: expr(xpr), skipper(s), attr(a), post_skip(skip_flag::postskip) {}

match_manip(Expr const& xpr, Skipper const& s
, BOOST_SCOPED_ENUM(skip_flag) ps, Attribute& a)
: expr(xpr), skipper(s), attr(a), post_skip(ps) {}

Expr expr;
Skipper const& skipper;
Attribute& attr;
BOOST_SCOPED_ENUM(skip_flag) const post_skip;

BOOST_DELETED_FUNCTION(match_manip& operator= (match_manip const&))
};

template <typename Expr, typename Enable = void>
struct match
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);
};

template <typename Expr>
struct match<Expr
, typename enable_if<traits::matches<qi::domain, Expr> >::type>
{
typedef match_manip<Expr> type;

static type call(Expr const& expr)
{
return type(expr, unused, unused);
}
};

template <typename Expr, typename Skipper, typename Enable = void>
struct phrase_match
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Expr);
};

template <typename Expr, typename Skipper>
struct phrase_match<Expr, Skipper
, typename enable_if<traits::matches<qi::domain, Expr> >::type>
{
typedef match_manip<Expr, mpl::false_, mpl::false_, Skipper> type;

static type call(
Expr const& expr
, Skipper const& skipper
, BOOST_SCOPED_ENUM(skip_flag) post_skip)
{
BOOST_SPIRIT_ASSERT_MATCH(qi::domain, Skipper);
return type(expr, skipper, post_skip, unused);
}
};

template<typename Char, typename Traits, typename Expr
, typename CopyExpr, typename CopyAttr>
inline std::basic_istream<Char, Traits> &
operator>>(std::basic_istream<Char, Traits> &is,
match_manip<Expr, CopyExpr, CopyAttr> const& fm)
{
typedef spirit::basic_istream_iterator<Char, Traits> input_iterator;

input_iterator f(is);
input_iterator l;
if (!qi::parse(f, l, fm.expr))
{
is.setstate(std::ios_base::failbit);
}
return is;
}

template<typename Char, typename Traits, typename Expr
, typename CopyExpr, typename CopyAttr
, typename Attribute>
inline std::basic_istream<Char, Traits> &
operator>>(std::basic_istream<Char, Traits> &is,
match_manip<Expr, CopyExpr, CopyAttr, unused_type, Attribute> const& fm)
{
typedef spirit::basic_istream_iterator<Char, Traits> input_iterator;

input_iterator f(is);
input_iterator l;
if (!qi::parse(f, l, fm.expr, fm.attr))
{
is.setstate(std::ios_base::failbit);
}
return is;
}

template<typename Char, typename Traits, typename Expr
, typename CopyExpr, typename CopyAttr
, typename Skipper>
inline std::basic_istream<Char, Traits> &
operator>>(std::basic_istream<Char, Traits> &is,
match_manip<Expr, CopyExpr, CopyAttr, Skipper> const& fm)
{
typedef spirit::basic_istream_iterator<Char, Traits> input_iterator;

input_iterator f(is);
input_iterator l;
if (!qi::phrase_parse(
f, l, fm.expr, fm.skipper, fm.post_skip))
{
is.setstate(std::ios_base::failbit);
}
return is;
}

template<typename Char, typename Traits, typename Expr
, typename CopyExpr, typename CopyAttr
, typename Attribute, typename Skipper
>
inline std::basic_istream<Char, Traits> &
operator>>(
std::basic_istream<Char, Traits> &is,
match_manip<Expr, CopyExpr, CopyAttr, Attribute, Skipper> const& fm)
{
typedef spirit::basic_istream_iterator<Char, Traits> input_iterator;

input_iterator f(is);
input_iterator l;
if (!qi::phrase_parse(
f, l, fm.expr, fm.skipper, fm.post_skip, fm.attr))
{
is.setstate(std::ios_base::failbit);
}
return is;
}

}}}}

#endif
