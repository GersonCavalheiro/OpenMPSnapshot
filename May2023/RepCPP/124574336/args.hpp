#if !defined(BOOST_PROTO_DONT_USE_PREPROCESSED_FILES)

#include <boost/proto/detail/preprocessed/args.hpp>

#elif !defined(BOOST_PP_IS_ITERATING)

#define BOOST_PROTO_DEFINE_CHILD_N(Z, N, DATA)                                              \
typedef BOOST_PP_CAT(Arg, N) BOOST_PP_CAT(child, N);                                    \


#define BOOST_PROTO_DEFINE_VOID_N(z, n, data)                                               \
typedef mpl::void_ BOOST_PP_CAT(child, n);                                              \


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/args.hpp")
#endif


#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

template< typename Arg0 >
struct term
{
static const long arity = 0;
typedef Arg0 child0;
BOOST_PP_REPEAT_FROM_TO(1, BOOST_PROTO_MAX_ARITY, BOOST_PROTO_DEFINE_VOID_N, ~)

typedef Arg0 back_;
};

#define BOOST_PP_ITERATION_PARAMS_1                                                         \
(3, (1, BOOST_PROTO_MAX_ARITY, <boost/proto/detail/args.hpp>))
#include BOOST_PP_ITERATE()

#if defined(__WAVE__) && defined(BOOST_PROTO_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#undef BOOST_PROTO_DEFINE_VOID_N
#undef BOOST_PROTO_DEFINE_CHILD_N

#else

#define N BOOST_PP_ITERATION()

template< BOOST_PP_ENUM_PARAMS(N, typename Arg) >
struct BOOST_PP_CAT(list, N)
{
static const long arity = N;
BOOST_PP_REPEAT(N, BOOST_PROTO_DEFINE_CHILD_N, ~)
BOOST_PP_REPEAT_FROM_TO(N, BOOST_PROTO_MAX_ARITY, BOOST_PROTO_DEFINE_VOID_N, ~)

typedef BOOST_PP_CAT(Arg, BOOST_PP_DEC(N)) back_;
};

#undef N

#endif
