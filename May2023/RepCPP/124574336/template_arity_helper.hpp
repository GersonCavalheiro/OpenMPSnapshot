#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#include <boost/proto/detail/preprocessed/template_arity_helper.hpp>

#elif !defined(BOOST_PP_IS_ITERATING)

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/template_arity_helper.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                                             \
(3, (1, BOOST_PROTO_MAX_ARITY, <boost/proto/detail/template_arity_helper.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#else

#define N BOOST_PP_ITERATION()

template<
template<BOOST_PP_ENUM_PARAMS(N, typename P)> class F
, BOOST_PP_ENUM_PARAMS(N, typename T)
>
sized_type<BOOST_PP_INC(N)>::type
template_arity_helper(F<BOOST_PP_ENUM_PARAMS(N, T)> **, mpl::int_<N> *);

#undef N

#endif 
