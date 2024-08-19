

#ifndef BOOST_BIMAP_DETAIL_DEBUG_STATIC_ERROR_HPP
#define BOOST_BIMAP_DETAIL_DEBUG_STATIC_ERROR_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>

#include <boost/mpl/assert.hpp>
#include <boost/preprocessor/cat.hpp>


#define BOOST_BIMAP_STATIC_ERROR(MESSAGE,VARIABLES)                           \
BOOST_MPL_ASSERT_MSG(false,                                           \
BOOST_PP_CAT(BIMAP_STATIC_ERROR__,MESSAGE),      \
VARIABLES)




#endif 
