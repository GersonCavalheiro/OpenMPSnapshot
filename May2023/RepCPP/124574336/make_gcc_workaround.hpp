#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#if BOOST_WORKAROUND(__GNUC__, == 3) || (BOOST_WORKAROUND(__GNUC__, == 4) && __GNUC_MINOR__ == 0)
#include <boost/proto/transform/detail/preprocessed/make_gcc_workaround.hpp>
#endif

#elif !defined(BOOST_PP_IS_ITERATING)

#define BOOST_PROTO_EXPR_MAKE_ARG(Z, M, DATA)                                                   \
detail::as_lvalue(                                                                          \
typename when<_, BOOST_PP_CAT(A, M)>::template impl<Expr, State, Data>()(e, s, d)       \
)                                                                                           \


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/make_gcc_workaround.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#if BOOST_WORKAROUND(__GNUC__, == 3) || (BOOST_WORKAROUND(__GNUC__, == 4) && __GNUC_MINOR__ == 0) || \
(defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES))

#define BOOST_PP_ITERATION_PARAMS_1                                                         \
(3, (0, BOOST_PROTO_MAX_ARITY, <boost/proto/transform/detail/make_gcc_workaround.hpp>))
#include BOOST_PP_ITERATE()

#endif

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#undef BOOST_PROTO_EXPR_MAKE_ARG

#else

#define N BOOST_PP_ITERATION()

template<typename Tag, typename Args, long Arity BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>
struct make<proto::expr<Tag, Args, Arity>(BOOST_PP_ENUM_PARAMS(N, A))>
: transform<make<proto::expr<Tag, Args, Arity>(BOOST_PP_ENUM_PARAMS(N, A))> >
{
template<typename Expr, typename State, typename Data>
struct impl : transform_impl<Expr, State, Data>
{
typedef proto::expr<Tag, Args, Arity> result_type;

BOOST_FORCEINLINE
result_type operator ()(
typename impl::expr_param   e
, typename impl::state_param  s
, typename impl::data_param   d
) const
{
return proto::expr<Tag, Args, Arity>::make(
BOOST_PP_ENUM(N, BOOST_PROTO_EXPR_MAKE_ARG, DATA)
);
}
};
};

template<typename Tag, typename Args, long Arity BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>
struct make<proto::basic_expr<Tag, Args, Arity>(BOOST_PP_ENUM_PARAMS(N, A))>
: transform<make<proto::basic_expr<Tag, Args, Arity>(BOOST_PP_ENUM_PARAMS(N, A))> >
{
template<typename Expr, typename State, typename Data>
struct impl : transform_impl<Expr, State, Data>
{
typedef proto::basic_expr<Tag, Args, Arity> result_type;

BOOST_FORCEINLINE
result_type operator ()(
typename impl::expr_param   e
, typename impl::state_param  s
, typename impl::data_param   d
) const
{
return proto::basic_expr<Tag, Args, Arity>::make(
BOOST_PP_ENUM(N, BOOST_PROTO_EXPR_MAKE_ARG, DATA)
);
}
};
};

#undef N

#endif
