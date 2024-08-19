#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#include <boost/proto/transform/detail/preprocessed/when.hpp>

#elif !defined(BOOST_PP_IS_ITERATING)

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/when.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

#define BOOST_PP_ITERATION_PARAMS_1                                                             \
(3, (0, BOOST_PROTO_MAX_ARITY, <boost/proto/transform/detail/when.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#else

#define N BOOST_PP_ITERATION()

template<typename Grammar, typename R BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>
struct when<Grammar, R(BOOST_PP_ENUM_PARAMS(N, A))>
: detail::when_impl<Grammar, R, R(BOOST_PP_ENUM_PARAMS(N, A))>
{};

#if N > 0
template<typename Grammar, typename R BOOST_PP_ENUM_TRAILING_PARAMS(N, typename A)>
struct when<Grammar, R(BOOST_PP_ENUM_PARAMS(N, A)...)>
: detail::when_impl<Grammar, R, R(BOOST_PP_ENUM_PARAMS(N, A)...)>
{};
#endif

#undef N

#endif
