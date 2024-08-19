

#ifndef BOOST_MULTI_INDEX_INDEXED_BY_HPP
#define BOOST_MULTI_INDEX_INDEXED_BY_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/mpl/vector.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/control/expr_if.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp> 





#if !defined(BOOST_MULTI_INDEX_LIMIT_INDEXED_BY_SIZE)
#define BOOST_MULTI_INDEX_LIMIT_INDEXED_BY_SIZE BOOST_MPL_LIMIT_VECTOR_SIZE
#endif

#if BOOST_MULTI_INDEX_LIMIT_INDEXED_BY_SIZE<BOOST_MPL_LIMIT_VECTOR_SIZE
#define BOOST_MULTI_INDEX_INDEXED_BY_SIZE \
BOOST_MULTI_INDEX_LIMIT_INDEXED_BY_SIZE
#else
#define BOOST_MULTI_INDEX_INDEXED_BY_SIZE BOOST_MPL_LIMIT_VECTOR_SIZE
#endif

#define BOOST_MULTI_INDEX_INDEXED_BY_TEMPLATE_PARM(z,n,var) \
typename BOOST_PP_CAT(var,n) BOOST_PP_EXPR_IF(n,=mpl::na)

namespace boost{

namespace multi_index{

template<
BOOST_PP_ENUM(
BOOST_MULTI_INDEX_INDEXED_BY_SIZE,
BOOST_MULTI_INDEX_INDEXED_BY_TEMPLATE_PARM,T)
>
struct indexed_by:
mpl::vector<BOOST_PP_ENUM_PARAMS(BOOST_MULTI_INDEX_INDEXED_BY_SIZE,T)>
{
};

} 

} 

#undef BOOST_MULTI_INDEX_INDEXED_BY_TEMPLATE_PARM
#undef BOOST_MULTI_INDEX_INDEXED_BY_SIZE

#endif
