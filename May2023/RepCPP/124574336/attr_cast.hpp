
#ifndef BOOST_SPIRIT_SUPPORT_AUXILIARY_ATTR_CAST_HPP
#define BOOST_SPIRIT_SUPPORT_AUXILIARY_ATTR_CAST_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/common_terminals.hpp>
#include <boost/spirit/home/support/attributes.hpp>
#include <boost/core/enable_if.hpp>
#include <boost/proto/traits.hpp>

namespace boost { namespace spirit
{
template <typename Expr>
typename enable_if<proto::is_expr<Expr>
, stateful_tag_type<Expr, tag::attr_cast> >::type
attr_cast(Expr const& expr)
{
return stateful_tag_type<Expr, tag::attr_cast>(expr);
}

template <typename Exposed, typename Expr>
typename enable_if<proto::is_expr<Expr>
, stateful_tag_type<Expr, tag::attr_cast, Exposed> >::type
attr_cast(Expr const& expr)
{
return stateful_tag_type<Expr, tag::attr_cast, Exposed>(expr);
}

template <typename Exposed, typename Transformed, typename Expr>
typename enable_if<proto::is_expr<Expr>
, stateful_tag_type<Expr, tag::attr_cast, Exposed, Transformed> >::type
attr_cast(Expr const& expr)
{
return stateful_tag_type<Expr, tag::attr_cast, Exposed, Transformed>(expr);
}
}}

#endif
