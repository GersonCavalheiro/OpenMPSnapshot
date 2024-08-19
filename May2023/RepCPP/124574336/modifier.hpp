
#ifndef BOOST_XPRESSIVE_DETAIL_STATIC_MODIFIER_HPP_EAN_10_04_2005
#define BOOST_XPRESSIVE_DETAIL_STATIC_MODIFIER_HPP_EAN_10_04_2005

#if defined(_MSC_VER)
# pragma once
# pragma warning(push)
# pragma warning(disable : 4510) 
# pragma warning(disable : 4610) 
#endif

#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/proto/traits.hpp>
#include <boost/xpressive/regex_constants.hpp>

namespace boost { namespace xpressive { namespace detail
{

template<typename Modifier>
struct modifier_op
{
typedef regex_constants::syntax_option_type opt_type;

template<typename Expr>
struct apply
{
typedef typename proto::binary_expr<
modifier_tag
, typename proto::terminal<Modifier>::type
, typename proto::result_of::as_child<Expr const>::type
>::type type;
};

template<typename Expr>
typename apply<Expr>::type const
operator ()(Expr const &expr) const
{
typename apply<Expr>::type that = {{this->mod_}, proto::as_child(expr)};
return that;
}

operator opt_type() const
{
return this->opt_;
}

Modifier mod_;
opt_type opt_;
};

}}}

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
