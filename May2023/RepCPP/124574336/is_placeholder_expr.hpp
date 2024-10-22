

#ifndef BOOST_FLYWEIGHT_DETAIL_IS_PLACEHOLDER_EXPR_HPP
#define BOOST_FLYWEIGHT_DETAIL_IS_PLACEHOLDER_EXPR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/type_traits/is_same.hpp>
#include <boost/mpl/apply.hpp>
#include <boost/mpl/aux_/lambda_support.hpp>
#include <boost/mpl/not.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>

namespace boost{

namespace flyweights{

namespace detail{



template<typename T>
struct is_placeholder_expression_helper
{
template<
BOOST_PP_ENUM_PARAMS(
BOOST_MPL_LIMIT_METAFUNCTION_ARITY,typename BOOST_PP_INTERCEPT)
>
struct apply{
typedef int type;
};

BOOST_MPL_AUX_LAMBDA_SUPPORT(1,is_placeholder_expression_helper,(T))
};

template<typename T>
struct is_placeholder_expression:
mpl::not_<is_same<
typename mpl::apply<
is_placeholder_expression_helper<T>,
BOOST_PP_ENUM_PARAMS(
BOOST_MPL_LIMIT_METAFUNCTION_ARITY,int BOOST_PP_INTERCEPT)
>::type,
int
> >
{};

} 

} 

} 

#endif
