
#if !defined(FUSION_TUPLE_FORWARD_10032005_0956)
#define FUSION_TUPLE_FORWARD_10032005_0956

#include <boost/fusion/support/config.hpp>
#include <boost/fusion/container/vector/detail/cpp03/limits.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>

#if !defined(BOOST_FUSION_DONT_USE_PREPROCESSED_FILES)
#include <boost/fusion/tuple/detail/preprocessed/tuple_fwd.hpp>
#else
#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/tuple" FUSION_MAX_VECTOR_SIZE_STR "_fwd.hpp")
#endif



#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

namespace boost { namespace fusion
{
struct void_;

template <
BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
FUSION_MAX_VECTOR_SIZE, typename T, void_)
>
struct tuple;
}}

#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#endif 

#endif

