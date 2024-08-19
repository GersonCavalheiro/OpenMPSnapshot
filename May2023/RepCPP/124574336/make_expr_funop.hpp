#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#include <boost/proto/detail/preprocessed/make_expr_funop.hpp>

#elif !defined(BOOST_PP_IS_ITERATING)

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/make_expr_funop.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                                     \
(3, (2, BOOST_PROTO_MAX_ARITY, <boost/proto/detail/make_expr_funop.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#else 

#define N BOOST_PP_ITERATION()

template<typename This BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>
struct result<This(BOOST_PP_ENUM_PARAMS(N, A))>
{
typedef
typename result_of::make_expr<
Tag
, Domain
BOOST_PP_ENUM_TRAILING_PARAMS(N, A)
>::type
type;
};

template<BOOST_PP_ENUM_PARAMS(N, typename A)>
BOOST_FORCEINLINE
typename result_of::make_expr<
Tag
, Domain
BOOST_PP_ENUM_TRAILING_PARAMS(N, const A)
>::type const
operator ()(BOOST_PP_ENUM_BINARY_PARAMS(N, const A, &a)) const
{
return proto::detail::make_expr_<
Tag
, Domain
BOOST_PP_ENUM_TRAILING_PARAMS(N, const A)
>()(BOOST_PP_ENUM_PARAMS(N, a));
}

#undef N

#endif
