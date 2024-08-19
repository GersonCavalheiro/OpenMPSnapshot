
#ifndef BOOST_XPRESSIVE_DETAIL_STATIC_TRANSFORMS_AS_SEQUENCE_HPP_EAN_04_01_2007
#define BOOST_XPRESSIVE_DETAIL_STATIC_TRANSFORMS_AS_SEQUENCE_HPP_EAN_04_01_2007

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/xpressive/detail/detail_fwd.hpp>
#include <boost/xpressive/detail/static/static.hpp>

namespace boost { namespace xpressive { namespace grammar_detail
{
template<typename Grammar, typename Callable = proto::callable>
struct in_sequence : proto::transform<in_sequence<Grammar, Callable> >
{
template<typename Expr, typename State, typename Data>
struct impl : proto::transform_impl<Expr, State, Data>
{
typedef
detail::static_xpression<
typename Grammar::template impl<Expr, State, Data>::result_type
, State
>
result_type;

result_type operator ()(
typename impl::expr_param expr
, typename impl::state_param state
, typename impl::data_param data
) const
{
return result_type(
typename Grammar::template impl<Expr, State, Data>()(expr, state, data)
, state
);
}
};
};

}}}

#endif
