
#if !defined(FUSION_SET_FORWARD_09162005_1102)
#define FUSION_SET_FORWARD_09162005_1102

#include <boost/fusion/support/config.hpp>
#include <boost/fusion/container/set/detail/cpp03/limits.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>

#if !defined(BOOST_FUSION_DONT_USE_PREPROCESSED_FILES)
#include <boost/fusion/container/set/detail/cpp03/preprocessed/set_fwd.hpp>
#else
#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/set" FUSION_MAX_SET_SIZE_STR "_fwd.hpp")
#endif



#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

namespace boost { namespace fusion
{
struct void_;
struct set_tag;
struct set_iterator_tag;

template <
BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
FUSION_MAX_SET_SIZE, typename T, void_)
>
struct set;
}}

#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#endif 

#endif
