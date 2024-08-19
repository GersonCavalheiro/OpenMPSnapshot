#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#include <boost/proto/detail/preprocessed/basic_expr.hpp>

#elif !defined(BOOST_PP_IS_ITERATING)

#define BOOST_PROTO_CHILD(Z, N, DATA)                                                       \
typedef BOOST_PP_CAT(Arg, N) BOOST_PP_CAT(proto_child, N);                              \
BOOST_PP_CAT(proto_child, N) BOOST_PP_CAT(child, N);                                    \


#define BOOST_PROTO_VOID(Z, N, DATA)                                                        \
typedef void BOOST_PP_CAT(proto_child, N);                                              \


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/basic_expr.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#define BOOST_PROTO_DEFINE_TERMINAL
#define BOOST_PP_ITERATION_PARAMS_1 (3, (0, 0, <boost/proto/detail/basic_expr.hpp>))
#include BOOST_PP_ITERATE()

#undef BOOST_PROTO_DEFINE_TERMINAL
#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, BOOST_PROTO_MAX_ARITY, <boost/proto/detail/basic_expr.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#undef BOOST_PROTO_CHILD
#undef BOOST_PROTO_VOID

#else

#define ARG_COUNT BOOST_PP_MAX(1, BOOST_PP_ITERATION())

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename Tag, typename Arg0>
struct basic_expr<Tag, term<Arg0>, 0>
#else
template<typename Tag BOOST_PP_ENUM_TRAILING_PARAMS(ARG_COUNT, typename Arg)>
struct basic_expr<Tag, BOOST_PP_CAT(list, BOOST_PP_ITERATION())<BOOST_PP_ENUM_PARAMS(ARG_COUNT, Arg)>, BOOST_PP_ITERATION() >
#endif
{
typedef Tag proto_tag;
static const long proto_arity_c = BOOST_PP_ITERATION();
typedef mpl::long_<BOOST_PP_ITERATION() > proto_arity;
typedef basic_expr proto_base_expr;
#ifdef BOOST_PROTO_DEFINE_TERMINAL
typedef term<Arg0> proto_args;
#else
typedef BOOST_PP_CAT(list, BOOST_PP_ITERATION())<BOOST_PP_ENUM_PARAMS(ARG_COUNT, Arg)> proto_args;
#endif
typedef basic_expr proto_grammar;
typedef basic_default_domain proto_domain;
typedef default_generator proto_generator;
typedef proto::tag::proto_expr<Tag, proto_domain> fusion_tag;
typedef basic_expr proto_derived_expr;
typedef void proto_is_expr_; 

BOOST_PP_REPEAT(ARG_COUNT, BOOST_PROTO_CHILD, ~)
BOOST_PP_REPEAT_FROM_TO(ARG_COUNT, BOOST_PROTO_MAX_ARITY, BOOST_PROTO_VOID, ~)

BOOST_FORCEINLINE
basic_expr const &proto_base() const
{
return *this;
}

BOOST_FORCEINLINE
basic_expr &proto_base()
{
return *this;
}

#ifdef BOOST_PROTO_DEFINE_TERMINAL
template<typename A0>
BOOST_FORCEINLINE
static basic_expr const make(A0 &a0)
{
return detail::make_terminal(a0, static_cast<basic_expr *>(0), static_cast<proto_args *>(0));
}

template<typename A0>
BOOST_FORCEINLINE
static basic_expr const make(A0 const &a0)
{
return detail::make_terminal(a0, static_cast<basic_expr *>(0), static_cast<proto_args *>(0));
}
#else
template<BOOST_PP_ENUM_PARAMS(ARG_COUNT, typename A)>
BOOST_FORCEINLINE
static basic_expr const make(BOOST_PP_ENUM_BINARY_PARAMS(ARG_COUNT, A, const &a))
{
basic_expr that = {BOOST_PP_ENUM_PARAMS(ARG_COUNT, a)};
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
};

#undef ARG_COUNT

#endif
