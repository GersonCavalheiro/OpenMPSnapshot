

#ifndef BOOST_MULTI_INDEX_TAG_HPP
#define BOOST_MULTI_INDEX_TAG_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/multi_index/detail/no_duplicate_tags.hpp>
#include <boost/mpl/identity.hpp>
#include <boost/mpl/transform.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/preprocessor/facilities/intercept.hpp> 
#include <boost/preprocessor/repetition/enum_binary_params.hpp> 
#include <boost/preprocessor/repetition/enum_params.hpp> 
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_and_derived.hpp>





#if !defined(BOOST_MULTI_INDEX_LIMIT_TAG_SIZE)
#define BOOST_MULTI_INDEX_LIMIT_TAG_SIZE BOOST_MPL_LIMIT_VECTOR_SIZE
#endif

#if BOOST_MULTI_INDEX_LIMIT_TAG_SIZE<BOOST_MPL_LIMIT_VECTOR_SIZE
#define BOOST_MULTI_INDEX_TAG_SIZE BOOST_MULTI_INDEX_LIMIT_TAG_SIZE
#else
#define BOOST_MULTI_INDEX_TAG_SIZE BOOST_MPL_LIMIT_VECTOR_SIZE
#endif

namespace boost{

namespace multi_index{

namespace detail{

struct tag_marker{};

template<typename T>
struct is_tag
{
BOOST_STATIC_CONSTANT(bool,value=(is_base_and_derived<tag_marker,T>::value));
};

} 

template<
BOOST_PP_ENUM_BINARY_PARAMS(
BOOST_MULTI_INDEX_TAG_SIZE,
typename T,
=mpl::na BOOST_PP_INTERCEPT) 
>
struct tag:private detail::tag_marker
{


typedef typename mpl::transform<
mpl::vector<BOOST_PP_ENUM_PARAMS(BOOST_MULTI_INDEX_TAG_SIZE,T)>,
mpl::identity<mpl::_1>
>::type type;

BOOST_STATIC_ASSERT(detail::no_duplicate_tags<type>::value);
};

} 

} 

#undef BOOST_MULTI_INDEX_TAG_SIZE

#endif
