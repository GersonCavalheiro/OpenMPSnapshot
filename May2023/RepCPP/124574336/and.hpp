
#ifndef BOOST_MPL_AND_HPP_INCLUDED
#define BOOST_MPL_AND_HPP_INCLUDED



#include <boost/mpl/aux_/config/use_preprocessed.hpp>

#if !defined(BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS) \
&& !defined(BOOST_MPL_PREPROCESSING_MODE)

#   include <boost/mpl/bool.hpp>
#   include <boost/mpl/aux_/nested_type_wknd.hpp>
#   include <boost/mpl/aux_/na_spec.hpp>
#   include <boost/mpl/aux_/lambda_support.hpp>

#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __GCCXML__
#if defined(and) 
#   pragma push_macro("and")
#   undef and
#   define and(x)
#endif
#endif
#endif

#   define BOOST_MPL_PREPROCESSED_HEADER and.hpp
#   include <boost/mpl/aux_/include_preprocessed.hpp>

#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __GCCXML__
#if defined(and) 
#   pragma pop_macro("and")
#endif
#endif
#endif

#else

#   define AUX778076_OP_NAME and_
#   define AUX778076_OP_VALUE1 false
#   define AUX778076_OP_VALUE2 true
#   include <boost/mpl/aux_/logical_op.hpp>

#endif 
#endif 
