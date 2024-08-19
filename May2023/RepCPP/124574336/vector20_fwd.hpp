#ifndef BOOST_PP_IS_ITERATING

#if !defined(BOOST_FUSION_VECTOR20_FWD_HPP_INCLUDED)
#define BOOST_FUSION_VECTOR20_FWD_HPP_INCLUDED

#include <boost/fusion/support/config.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

#if !defined(BOOST_FUSION_DONT_USE_PREPROCESSED_FILES)
#include <boost/fusion/container/vector/detail/cpp03/preprocessed/vector20_fwd.hpp>
#else
#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 2, line: 0, output: "preprocessed/vector20_fwd.hpp")
#endif



#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(preserve: 1)
#endif

namespace boost { namespace fusion
{
#define BOOST_PP_FILENAME_1 <boost/fusion/container/vector/detail/cpp03/vector20_fwd.hpp>
#define BOOST_PP_ITERATION_LIMITS (11, 20)
#include BOOST_PP_ITERATE()
}}

#if defined(__WAVE__) && defined(BOOST_FUSION_CREATE_PREPROCESSED_FILES)
#pragma wave option(output: null)
#endif

#endif 

#endif

#else

template <BOOST_PP_ENUM_PARAMS(BOOST_PP_ITERATION(), typename T)>
struct BOOST_PP_CAT(vector, BOOST_PP_ITERATION());

#endif
