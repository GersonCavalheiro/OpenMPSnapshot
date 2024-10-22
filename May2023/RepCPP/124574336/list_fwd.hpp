
#if !defined(FUSION_LIST_FORWARD_07172005_0224)
#define FUSION_LIST_FORWARD_07172005_0224

#include <boost/fusion/support/config.hpp>
#include <boost/fusion/container/list/detail/cpp03/limits.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>

#if !defined(BOOST_FUSION_DONT_USE_PREPROCESSED_FILES)
#include <boost/fusion/container/list/detail/cpp03/preprocessed/list_fwd.hpp>
#else
#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/list" FUSION_MAX_LIST_SIZE_STR "_fwd.hpp")
#endif



#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

namespace boost { namespace fusion
{
struct void_;

template <
BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
FUSION_MAX_LIST_SIZE, typename T, void_)
>
struct list;
}}

#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#endif 

#endif
