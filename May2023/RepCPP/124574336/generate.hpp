
#if !defined(BOOST_SPIRIT_KARMA_DETAIL_GENERATE_FEB_20_2007_0959AM)
#define BOOST_SPIRIT_KARMA_DETAIL_GENERATE_FEB_20_2007_0959AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/karma/meta_compiler.hpp>
#include <boost/spirit/home/karma/delimit_out.hpp>
#include <boost/spirit/home/karma/delimit_flag.hpp>
#include <boost/spirit/home/karma/detail/output_iterator.hpp>
#include <boost/spirit/home/support/unused.hpp>
#include <boost/mpl/assert.hpp>
#include <boost/mpl/bool.hpp>

namespace boost { namespace spirit { namespace karma { namespace detail
{
template <typename Expr, typename Enable = void>
struct generate_impl
{
BOOST_SPIRIT_ASSERT_MATCH(karma::domain, Expr);
};

template <typename Expr>
struct generate_impl<Expr
, typename enable_if<traits::matches<karma::domain, Expr> >::type>
{
template <typename OutputIterator>
static bool call(
OutputIterator& target_sink
, Expr const& expr)
{
typedef traits::properties_of<
typename result_of::compile<karma::domain, Expr>::type
> properties;

output_iterator<OutputIterator
, mpl::int_<properties::value> > sink(target_sink);
return compile<karma::domain>(expr).
generate(sink, unused, unused, unused);
}

template <typename OutputIterator, typename Properties>
static bool call(
detail::output_iterator<OutputIterator, Properties>& sink
, Expr const& expr)
{
return compile<karma::domain>(expr).
generate(sink, unused, unused, unused);
}
};

template <typename Expr, typename Enable = void>
struct generate_delimited_impl
{
BOOST_SPIRIT_ASSERT_MATCH(karma::domain, Expr);
};

template <typename Expr>
struct generate_delimited_impl<Expr
, typename enable_if<traits::matches<karma::domain, Expr> >::type>
{
template <typename OutputIterator, typename Delimiter>
static bool call(
OutputIterator& target_sink
, Expr const& expr
, Delimiter const& delimiter
, BOOST_SCOPED_ENUM(delimit_flag) pre_delimit)
{
typedef traits::properties_of<
typename result_of::compile<karma::domain, Expr>::type
> properties;
typedef traits::properties_of<
typename result_of::compile<karma::domain, Delimiter>::type
> delimiter_properties;

detail::output_iterator<OutputIterator
, mpl::int_<properties::value | delimiter_properties::value>
> sink(target_sink);
return call(sink, expr, delimiter, pre_delimit);
}

template <typename OutputIterator, typename Properties
, typename Delimiter>
static bool call(
detail::output_iterator<OutputIterator, Properties>& sink
, Expr const& expr
, Delimiter const& delimiter
, BOOST_SCOPED_ENUM(delimit_flag) pre_delimit)
{
BOOST_SPIRIT_ASSERT_MATCH(karma::domain, Delimiter);

typename result_of::compile<karma::domain, Delimiter>::type const 
delimiter_ = compile<karma::domain>(delimiter);

if (pre_delimit == delimit_flag::predelimit &&
!karma::delimit_out(sink, delimiter_))
{
return false;
}
return compile<karma::domain>(expr).
generate(sink, unused, delimiter_, unused);
}
};

}}}}

#endif

