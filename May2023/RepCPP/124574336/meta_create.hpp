
#if !defined(BOOST_SPIRIT_SUPPORT_META_CREATE_NOV_21_2009_0327PM)
#define BOOST_SPIRIT_SUPPORT_META_CREATE_NOV_21_2009_0327PM

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/spirit/home/support/unused.hpp>

#include <boost/version.hpp>
#include <boost/proto/make_expr.hpp>
#include <boost/proto/traits.hpp>
#include <boost/utility/result_of.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/remove_reference.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>

#if defined(__GNUC__) && ((__GNUC__ < 4) || (__GNUC__ == 4) && (__GNUC_MINOR__ < 3))
#include <boost/type_traits/is_same.hpp>
#endif

namespace boost { namespace spirit { namespace traits
{
template <typename Domain, typename T, typename Enable = void>
struct meta_create;

template <typename Domain, typename T, typename Enable = void>
struct meta_create_exists : mpl::false_ {};
}}}

namespace boost { namespace spirit 
{
namespace detail
{
template <typename T>
struct add_const_ref
: add_reference<typename add_const<T>::type> {};

template <typename T>
struct remove_const_ref
: remove_const<typename remove_reference<T>::type> {};

#if BOOST_VERSION < 104200
template <typename OpTag, typename Domain>
struct nary_proto_expr_function
{
template <typename T>
struct result;

#if defined(__GNUC__) && ((__GNUC__ < 4) || (__GNUC__ == 4) && (__GNUC_MINOR__ < 3))
template <typename F, typename T1, typename T2>
struct result<F(T1, T2)>
{
BOOST_STATIC_ASSERT((is_same<F, nary_proto_expr_function>::value));
#else
template <typename T1, typename T2>
struct result<nary_proto_expr_function(T1, T2)>
{
#endif
typedef typename remove_const_ref<T2>::type left_type;
typedef typename 
spirit::traits::meta_create<Domain, T1>::type
right_type;

typedef typename mpl::eval_if<
traits::not_is_unused<left_type>
, proto::result_of::make_expr<OpTag, left_type, right_type>
, mpl::identity<right_type>
>::type type;
};

template <typename T>
typename result<nary_proto_expr_function(T, unused_type const&)>::type
operator()(T, unused_type const&) const
{
typedef spirit::traits::meta_create<Domain, T> right_type;
return right_type::call();
}

template <typename T1, typename T2>
typename result<nary_proto_expr_function(T1, T2)>::type
operator()(T1, T2 const& t2) const
{
typedef spirit::traits::meta_create<Domain, T1> right_type;
return proto::make_expr<OpTag>(t2, right_type::call());
}
};
#else
template <typename OpTag, typename Domain>
struct nary_proto_expr_function
{
template <typename T>
struct result;

#if defined(__GNUC__) && ((__GNUC__ < 4) || (__GNUC__ == 4) && (__GNUC_MINOR__ < 3))
template <typename F, typename T1, typename T2>
struct result<F(T1, T2)>
{
BOOST_STATIC_ASSERT((is_same<F, nary_proto_expr_function>::value));
#else
template <typename T1, typename T2>
struct result<nary_proto_expr_function(T1, T2)>
{
#endif
typedef typename remove_const_ref<T1>::type left_type;
typedef typename 
spirit::traits::meta_create<Domain, T2>::type
right_type;

typedef typename mpl::eval_if<
traits::not_is_unused<left_type>
, proto::result_of::make_expr<OpTag, left_type, right_type>
, mpl::identity<right_type>
>::type type;
};

template <typename T>
typename result<nary_proto_expr_function(unused_type const&, T)>::type
operator()(unused_type const&, T) const
{
typedef spirit::traits::meta_create<Domain, T> right_type;
return right_type::call();
}

template <typename T1, typename T2>
typename result<nary_proto_expr_function(T1, T2)>::type
operator()(T1 const& t1, T2) const
{
typedef spirit::traits::meta_create<Domain, T2> right_type;
return proto::make_expr<OpTag>(t1, right_type::call());
}
};
#endif
}

template <typename T, typename OpTag, typename Domain>
struct make_unary_proto_expr
{
typedef spirit::traits::meta_create<Domain, T> subject_type;

typedef typename proto::result_of::make_expr<
OpTag, typename subject_type::type
>::type type;

static type call()
{
return proto::make_expr<OpTag>(subject_type::call());
}
};

template <typename Sequence, typename OpTag, typename Domain>
struct make_nary_proto_expr
{
typedef detail::nary_proto_expr_function<OpTag, Domain> 
make_proto_expr;

typedef typename fusion::result_of::fold<
Sequence, unused_type, make_proto_expr
>::type type;

static type call()
{
return fusion::fold(Sequence(), unused, make_proto_expr());
}
};

namespace detail
{
template <typename T>
struct is_fusion_sequence_but_not_proto_expr
: mpl::and_<
fusion::traits::is_sequence<T>
, mpl::not_<proto::is_expr<T> > >
{};
}
}}

#endif
