#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#include <boost/proto/detail/preprocessed/expr_variadic.hpp>
#else
#include <boost/proto/detail/preprocessed/expr.hpp>
#endif

#elif !defined(BOOST_PP_IS_ITERATING)

#define BOOST_PROTO_CHILD(Z, N, DATA)                                                       \
typedef BOOST_PP_CAT(Arg, N) BOOST_PP_CAT(proto_child, N);                              \
BOOST_PP_CAT(proto_child, N) BOOST_PP_CAT(child, N);                                    \


#define BOOST_PROTO_VOID(Z, N, DATA)                                                        \
typedef void BOOST_PP_CAT(proto_child, N);                                              \


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/expr_variadic.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#define BOOST_PROTO_DEFINE_TERMINAL
#define BOOST_PP_ITERATION_PARAMS_1                                                         \
(3, (0, 0, <boost/proto/detail/expr.hpp>))
#include BOOST_PP_ITERATE()

#undef BOOST_PROTO_DEFINE_TERMINAL
#define BOOST_PP_ITERATION_PARAMS_1                                                         \
(3, (1, BOOST_PROTO_MAX_ARITY, <boost/proto/detail/expr.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/expr.hpp")


#pragma wave option(preserve: 1)

#define BOOST_PROTO_DEFINE_TERMINAL
#define BOOST_PP_ITERATION_PARAMS_1 (3, (0, 0, <boost/proto/detail/expr.hpp>))
#include BOOST_PP_ITERATE()

#undef BOOST_PROTO_DEFINE_TERMINAL
#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, BOOST_PROTO_MAX_ARITY, <boost/proto/detail/expr.hpp>))
#include BOOST_PP_ITERATE()

#pragma wave option(output: null)
#undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif

#undef BOOST_PROTO_CHILD
#undef BOOST_PROTO_VOID

#else

#define ARG_COUNT BOOST_PP_MAX(1, BOOST_PP_ITERATION())

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename Tag, typename Arg0>
struct expr<Tag, term<Arg0>, 0>
#else
template<typename Tag BOOST_PP_ENUM_TRAILING_PARAMS(ARG_COUNT, typename Arg)>
struct expr<Tag, BOOST_PP_CAT(list, BOOST_PP_ITERATION())<BOOST_PP_ENUM_PARAMS(ARG_COUNT, Arg)>, BOOST_PP_ITERATION() >
#endif
{
typedef Tag proto_tag;
static const long proto_arity_c = BOOST_PP_ITERATION();
typedef mpl::long_<BOOST_PP_ITERATION() > proto_arity;
typedef expr proto_base_expr;
#ifdef BOOST_PROTO_DEFINE_TERMINAL
typedef term<Arg0> proto_args;
#else
typedef BOOST_PP_CAT(list, BOOST_PP_ITERATION())<BOOST_PP_ENUM_PARAMS(ARG_COUNT, Arg)> proto_args;
#endif
typedef basic_expr<Tag, proto_args, BOOST_PP_ITERATION() > proto_grammar;
typedef default_domain proto_domain;
typedef default_generator proto_generator;
typedef proto::tag::proto_expr<Tag, proto_domain> fusion_tag;
typedef expr proto_derived_expr;
typedef void proto_is_expr_; 

BOOST_PP_REPEAT(ARG_COUNT, BOOST_PROTO_CHILD, ~)
BOOST_PP_REPEAT_FROM_TO(ARG_COUNT, BOOST_PROTO_MAX_ARITY, BOOST_PROTO_VOID, ~)

BOOST_FORCEINLINE
expr const &proto_base() const
{
return *this;
}

BOOST_FORCEINLINE
expr &proto_base()
{
return *this;
}

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename A0>
BOOST_FORCEINLINE
static expr const make(A0 &a0)
{
return detail::make_terminal(a0, static_cast<expr *>(0), static_cast<proto_args *>(0));
}

template<typename A0>
BOOST_FORCEINLINE
static expr const make(A0 const &a0)
{
return detail::make_terminal(a0, static_cast<expr *>(0), static_cast<proto_args *>(0));
}
#else
template<BOOST_PP_ENUM_PARAMS(ARG_COUNT, typename A)>
BOOST_FORCEINLINE
static expr const make(BOOST_PP_ENUM_BINARY_PARAMS(ARG_COUNT, A, const &a))
{
expr that = {BOOST_PP_ENUM_PARAMS(ARG_COUNT, a)};
return that;
}
#endif

#if 1 == BOOST_PP_ITERATION()
typedef typename detail::address_of_hack<Tag, proto_child0>::type address_of_hack_type_;

BOOST_FORCEINLINE
operator address_of_hack_type_() const
{
return boost::addressof(this->child0);
}
#else
typedef detail::not_a_valid_type address_of_hack_type_;
#endif

BOOST_FORCEINLINE
proto::expr<
proto::tag::assign
, list2<expr &, expr const &>
, 2
> const
operator =(expr const &a)
{
proto::expr<
proto::tag::assign
, list2<expr &, expr const &>
, 2
> that = {*this, a};
return that;
}

template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::assign
, list2<expr const &, typename result_of::as_child<A>::type>
, 2
> const
operator =(A &a) const
{
proto::expr<
proto::tag::assign
, list2<expr const &, typename result_of::as_child<A>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}

template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::assign
, list2<expr const &, typename result_of::as_child<A const>::type>
, 2
> const
operator =(A const &a) const
{
proto::expr<
proto::tag::assign
, list2<expr const &, typename result_of::as_child<A const>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::assign
, list2<expr &, typename result_of::as_child<A>::type>
, 2
> const
operator =(A &a)
{
proto::expr<
proto::tag::assign
, list2<expr &, typename result_of::as_child<A>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}

template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::assign
, list2<expr &, typename result_of::as_child<A const>::type>
, 2
> const
operator =(A const &a)
{
proto::expr<
proto::tag::assign
, list2<expr &, typename result_of::as_child<A const>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}
#endif

template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::subscript
, list2<expr const &, typename result_of::as_child<A>::type>
, 2
> const
operator [](A &a) const
{
proto::expr<
proto::tag::subscript
, list2<expr const &, typename result_of::as_child<A>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}

template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::subscript
, list2<expr const &, typename result_of::as_child<A const>::type>
, 2
> const
operator [](A const &a) const
{
proto::expr<
proto::tag::subscript
, list2<expr const &, typename result_of::as_child<A const>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::subscript
, list2<expr &, typename result_of::as_child<A>::type>
, 2
> const
operator [](A &a)
{
proto::expr<
proto::tag::subscript
, list2<expr &, typename result_of::as_child<A>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}

template<typename A>
BOOST_FORCEINLINE
proto::expr<
proto::tag::subscript
, list2<expr &, typename result_of::as_child<A const>::type>
, 2
> const
operator [](A const &a)
{
proto::expr<
proto::tag::subscript
, list2<expr &, typename result_of::as_child<A const>::type>
, 2
> that = {*this, proto::as_child(a)};
return that;
}
#endif

template<typename Sig>
struct result
{
typedef typename result_of::funop<Sig, expr, default_domain>::type const type;
};

#ifndef BOOST_NO_CXX11_VARIADIC_TEMPLATES
template<typename ...A>
BOOST_FORCEINLINE
typename result_of::funop<
expr const(A const &...)
, expr
, default_domain
>::type const
operator ()(A const &... a) const
{
return result_of::funop<
expr const(A const &...)
, expr
, default_domain
>::call(*this, a...);
}

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename ...A>
BOOST_FORCEINLINE
typename result_of::funop<
expr(A const &...)
, expr
, default_domain
>::type const
operator ()(A const &... a)
{
return result_of::funop<
expr(A const &...)
, expr
, default_domain
>::call(*this, a...);
}
#endif

#else 

BOOST_FORCEINLINE
proto::expr<proto::tag::function, list1<expr const &>, 1> const
operator ()() const
{
proto::expr<proto::tag::function, list1<expr const &>, 1> that = {*this};
return that;
}

#ifdef BOOST_PROTO_DEFINE_TERMINAL
BOOST_FORCEINLINE
proto::expr<proto::tag::function, list1<expr &>, 1> const
operator ()()
{
proto::expr<proto::tag::function, list1<expr &>, 1> that = {*this};
return that;
}
#endif

#define BOOST_PP_ITERATION_PARAMS_2                                                             \
(3, (1, BOOST_PP_DEC(BOOST_PROTO_MAX_FUNCTION_CALL_ARITY), <boost/proto/detail/expr_funop.hpp>))
#include BOOST_PP_ITERATE()

#endif
};

#undef ARG_COUNT

#endif
