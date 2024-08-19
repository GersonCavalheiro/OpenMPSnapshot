

#ifndef BOOST_FLYWEIGHT_DETAIL_NOT_PLACEHOLDER_EXPR_HPP
#define BOOST_FLYWEIGHT_DETAIL_NOT_PLACEHOLDER_EXPR_HPP

#if defined(_MSC_VER)
#pragma once
#endif



#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>

#if BOOST_WORKAROUND(__GNUC__, <4)||\
BOOST_WORKAROUND(__GNUC__,==4)&&(__GNUC_MINOR__<2)||\
BOOST_WORKAROUND(__GNUC__, ==7)&&( __cplusplus>=201703L)||\
BOOST_WORKAROUND(__GNUC__, >=8)&&( __cplusplus>=201103L)


#include <boost/mpl/limits/arity.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_trailing_params.hpp>

#define BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION                  \
BOOST_PP_ENUM_TRAILING_PARAMS(                                        \
BOOST_MPL_LIMIT_METAFUNCTION_ARITY,typename=int BOOST_PP_INTERCEPT)
#define BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION_DEF              \
BOOST_PP_ENUM_TRAILING_PARAMS(                                        \
BOOST_MPL_LIMIT_METAFUNCTION_ARITY,typename BOOST_PP_INTERCEPT)

#else
#define BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION  ,int=0
#define BOOST_FLYWEIGHT_NOT_A_PLACEHOLDER_EXPRESSION_DEF  ,int
#endif

#endif
