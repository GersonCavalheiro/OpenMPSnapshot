#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#include <boost/proto/detail/preprocessed/and_n.hpp>

#elif !defined(BOOST_PP_IS_ITERATING)

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/and_n.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                                             \
(3, (2, BOOST_PP_MAX(BOOST_PROTO_MAX_ARITY, BOOST_PROTO_MAX_LOGICAL_ARITY), <boost/proto/detail/and_n.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#else 

#define N BOOST_PP_ITERATION()

template<bool B, BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename P)>
struct BOOST_PP_CAT(and_, N)
#if 2 == N
: mpl::bool_<P0::value>
{};
#else
: BOOST_PP_CAT(and_, BOOST_PP_DEC(N))<
P0::value BOOST_PP_COMMA_IF(BOOST_PP_SUB(N,2))
BOOST_PP_ENUM_SHIFTED_PARAMS(BOOST_PP_DEC(N), P)
>
{};
#endif

template<BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), typename P)>
struct BOOST_PP_CAT(and_, N)<false, BOOST_PP_ENUM_PARAMS(BOOST_PP_DEC(N), P)>
: mpl::false_
{};

#if N <= BOOST_PROTO_MAX_LOGICAL_ARITY

template<BOOST_PP_ENUM_PARAMS(N, typename G), typename Expr, typename State, typename Data>
struct _and_impl<proto::and_<BOOST_PP_ENUM_PARAMS(N, G)>, Expr, State, Data>
: proto::transform_impl<Expr, State, Data>
{
#define M0(Z, N, DATA)                                                            \
typedef                                                                           \
typename proto::when<proto::_, BOOST_PP_CAT(G, N)>                            \
::template impl<Expr, State, Data>                                        \
BOOST_PP_CAT(Gimpl, N);                                                           \

BOOST_PP_REPEAT(N, M0, ~)
#undef M0

typedef typename BOOST_PP_CAT(Gimpl, BOOST_PP_DEC(N))::result_type result_type;

result_type operator()(
typename _and_impl::expr_param e
, typename _and_impl::state_param s
, typename _and_impl::data_param d
) const
{
#define M0(Z,N,DATA) BOOST_PP_CAT(Gimpl,N)()(e,s,d);
BOOST_PP_REPEAT(BOOST_PP_DEC(N),M0,~)
return BOOST_PP_CAT(Gimpl,BOOST_PP_DEC(N))()(e,s,d);
#undef M0
}
};

#endif

#undef N

#endif
