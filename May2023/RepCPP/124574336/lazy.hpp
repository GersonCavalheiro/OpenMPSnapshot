
#if !defined(BOOST_SPIRIT_LAZY_NOVEMBER_04_2008_1157AM)
#define BOOST_SPIRIT_LAZY_NOVEMBER_04_2008_1157AM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/modify.hpp>
#include <boost/spirit/home/support/detail/is_spirit_tag.hpp>
#include <boost/proto/traits.hpp>

namespace boost { namespace phoenix
{
template <typename Expr>
struct actor;
}}

namespace boost { namespace spirit
{
template <typename Eval>
typename proto::terminal<phoenix::actor<Eval> >::type
lazy(phoenix::actor<Eval> const& f)
{
return proto::terminal<phoenix::actor<Eval> >::type::make(f);
}

namespace tag
{
struct lazy_eval 
{
BOOST_SPIRIT_IS_TAG()
};
}

template <typename Domain>
struct is_modifier_directive<Domain, tag::lazy_eval>
: mpl::true_ {};
}}

#endif
