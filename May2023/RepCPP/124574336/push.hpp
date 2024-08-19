

#ifndef BOOST_IOSTREAMS_DETAIL_PUSH_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_PUSH_HPP_INCLUDED 

#if defined(_MSC_VER)
# pragma once
#endif                    

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/categories.hpp>
#include <boost/iostreams/detail/adapter/range_adapter.hpp>
#include <boost/iostreams/detail/config/wide_streams.hpp>
#include <boost/iostreams/detail/enable_if_stream.hpp>   
#include <boost/iostreams/pipeline.hpp>   
#include <boost/iostreams/detail/push_params.hpp>   
#include <boost/iostreams/detail/resolve.hpp>
#include <boost/mpl/bool.hpp>   
#include <boost/preprocessor/cat.hpp> 
#include <boost/preprocessor/control/iif.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_convertible.hpp>

#define BOOST_IOSTREAMS_DEFINE_PUSH_CONSTRUCTOR(name, mode, ch, helper) \
BOOST_IOSTREAMS_DEFINE_PUSH_IMPL(name, mode, ch, helper, 0, ?) \


#define BOOST_IOSTREAMS_DEFINE_PUSH(name, mode, ch, helper) \
BOOST_IOSTREAMS_DEFINE_PUSH_IMPL(name, mode, ch, helper, 1, void) \



#define BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, arg, helper, has_return) \
this->helper( ::boost::iostreams::detail::resolve<mode, ch>(arg) \
BOOST_IOSTREAMS_PUSH_ARGS() ); \


#if !BOOST_WORKAROUND(BOOST_BORLANDC, < 0x600) \

# ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
#  define BOOST_IOSTREAMS_DEFINE_PUSH_IMPL(name, mode, ch, helper, has_return, result) \
template<typename CharType, typename TraitsType> \
BOOST_PP_IIF(has_return, result, explicit) \
name(::std::basic_streambuf<CharType, TraitsType>& sb BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, sb, helper, has_return); } \
template<typename CharType, typename TraitsType> \
BOOST_PP_IIF(has_return, result, explicit) \
name(::std::basic_istream<CharType, TraitsType>& is BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_STATIC_ASSERT((!is_convertible<mode, output>::value)); \
BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, is, helper, has_return); } \
template<typename CharType, typename TraitsType> \
BOOST_PP_IIF(has_return, result, explicit) \
name(::std::basic_ostream<CharType, TraitsType>& os BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_STATIC_ASSERT((!is_convertible<mode, input>::value)); \
BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, os, helper, has_return); } \
template<typename CharType, typename TraitsType> \
BOOST_PP_IIF(has_return, result, explicit) \
name(::std::basic_iostream<CharType, TraitsType>& io BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, io, helper, has_return); } \
template<typename Iter> \
BOOST_PP_IIF(has_return, result, explicit) \
name(const iterator_range<Iter>& rng BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_PP_EXPR_IF(has_return, return) \
this->helper( ::boost::iostreams::detail::range_adapter< \
mode, iterator_range<Iter> \
>(rng) \
BOOST_IOSTREAMS_PUSH_ARGS() ); } \
template<typename Pipeline, typename Concept> \
BOOST_PP_IIF(has_return, result, explicit) \
name(const ::boost::iostreams::pipeline<Pipeline, Concept>& p) \
{ p.push(*this); } \
template<typename T> \
BOOST_PP_IIF(has_return, result, explicit) \
name(const T& t BOOST_IOSTREAMS_PUSH_PARAMS() BOOST_IOSTREAMS_DISABLE_IF_STREAM(T)) \
{ this->helper( ::boost::iostreams::detail::resolve<mode, ch>(t) \
BOOST_IOSTREAMS_PUSH_ARGS() ); } \

# else 
#  define BOOST_IOSTREAMS_DEFINE_PUSH_IMPL(name, mode, ch, helper, has_return, result) \
BOOST_PP_IF(has_return, result, explicit) \
name(::std::streambuf& sb BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, sb, helper, has_return); } \
BOOST_PP_IF(has_return, result, explicit) \
name(::std::istream& is BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_STATIC_ASSERT((!is_convertible<mode, output>::value)); \
BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, is, helper, has_return); } \
BOOST_PP_IF(has_return, result, explicit) \
name(::std::ostream& os BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_STATIC_ASSERT((!is_convertible<mode, input>::value)); \
BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, os, helper, has_return); } \
BOOST_PP_IF(has_return, result, explicit) \
name(::std::iostream& io BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_IOSTREAMS_ADAPT_STREAM(mode, ch, io, helper, has_return); } \
template<typename Iter> \
BOOST_PP_IF(has_return, result, explicit) \
name(const iterator_range<Iter>& rng BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ BOOST_PP_EXPR_IF(has_return, return) \
this->helper( ::boost::iostreams::detail::range_adapter< \
mode, iterator_range<Iter> \
>(rng) \
BOOST_IOSTREAMS_PUSH_ARGS() ); } \
template<typename Pipeline, typename Concept> \
BOOST_PP_IF(has_return, result, explicit) \
name(const ::boost::iostreams::pipeline<Pipeline, Concept>& p) \
{ p.push(*this); } \
template<typename T> \
BOOST_PP_EXPR_IF(has_return, result) \
name(const T& t BOOST_IOSTREAMS_PUSH_PARAMS() BOOST_IOSTREAMS_DISABLE_IF_STREAM(T)) \
{ this->helper( ::boost::iostreams::detail::resolve<mode, ch>(t) \
BOOST_IOSTREAMS_PUSH_ARGS() ); } \

# endif 
#else 
# define BOOST_IOSTREAMS_DEFINE_PUSH_IMPL(name, mode, ch, helper, has_return, result) \
template<typename T> \
void BOOST_PP_CAT(name, _msvc_impl) \
( ::boost::mpl::true_, const T& t BOOST_IOSTREAMS_PUSH_PARAMS() ) \
{ t.push(*this); } \
template<typename T> \
void BOOST_PP_CAT(name, _msvc_impl) \
( ::boost::mpl::false_, const T& t BOOST_IOSTREAMS_PUSH_PARAMS() ) \
{ this->helper( ::boost::iostreams::detail::resolve<mode, ch>(t) \
BOOST_IOSTREAMS_PUSH_ARGS() ); } \
template<typename T> \
BOOST_PP_IF(has_return, result, explicit) \
name(const T& t BOOST_IOSTREAMS_PUSH_PARAMS()) \
{ \
this->BOOST_PP_CAT(name, _msvc_impl) \
( ::boost::iostreams::detail::is_pipeline<T>(), \
t BOOST_IOSTREAMS_PUSH_ARGS() ); \
} \

#endif 

#endif 
