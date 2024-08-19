
#ifndef BOOST_PROTO_TRANSFORM_PASS_THROUGH_HPP_EAN_12_26_2006
#define BOOST_PROTO_TRANSFORM_PASS_THROUGH_HPP_EAN_12_26_2006

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/proto/proto_fwd.hpp>
#include <boost/proto/args.hpp>
#include <boost/proto/transform/impl.hpp>
#include <boost/proto/detail/ignore_unused.hpp>

#if defined(_MSC_VER)
# pragma warning(push)
# pragma warning(disable : 4714) 
#endif

namespace boost { namespace proto
{
namespace detail
{
template<
typename Grammar
, typename Domain
, typename Expr
, typename State
, typename Data
, long Arity = arity_of<Expr>::value
>
struct pass_through_impl
{};

#include <boost/proto/transform/detail/pass_through_impl.hpp>

template<typename Grammar, typename Domain, typename Expr, typename State, typename Data>
struct pass_through_impl<Grammar, Domain, Expr, State, Data, 0>
: transform_impl<Expr, State, Data>
{
typedef Expr result_type;

BOOST_FORCEINLINE
BOOST_PROTO_RETURN_TYPE_STRICT_LOOSE(result_type, typename pass_through_impl::expr_param)
operator()(
typename pass_through_impl::expr_param e
, typename pass_through_impl::state_param
, typename pass_through_impl::data_param
) const
{
return e;
}
};

} 

template<typename Grammar, typename Domain >
struct pass_through
: transform<pass_through<Grammar, Domain> >
{
template<typename Expr, typename State, typename Data>
struct impl
: detail::pass_through_impl<Grammar, Domain, Expr, State, Data>
{};
};

template<typename Grammar, typename Domain>
struct is_callable<pass_through<Grammar, Domain> >
: mpl::true_
{};

}} 

#if defined(_MSC_VER)
# pragma warning(pop)
#endif

#endif
