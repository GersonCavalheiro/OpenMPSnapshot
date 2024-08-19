
#if !defined(FUSION_PP_DEQUE_FORWARD_02092007_0749)
#define FUSION_PP_DEQUE_FORWARD_02092007_0749

#if defined(BOOST_FUSION_HAS_VARIADIC_DEQUE)
#error "C++03 only! This file should not have been included"
#endif

#include <boost/fusion/container/deque/detail/cpp03/limits.hpp>
#include <boost/preprocessor/repetition/enum_params_with_a_default.hpp>

#if !defined(BOOST_FUSION_DONT_USE_PREPROCESSED_FILES)
#include <boost/fusion/container/deque/detail/cpp03/preprocessed/deque_fwd.hpp>
#else
#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/deque" FUSION_MAX_DEQUE_SIZE_STR "_fwd.hpp")
#endif



#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

namespace boost { namespace fusion
{
struct void_;

template<
BOOST_PP_ENUM_PARAMS_WITH_A_DEFAULT(
FUSION_MAX_DEQUE_SIZE, typename T, void_)>
struct deque;
}}

#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#endif 

#endif
