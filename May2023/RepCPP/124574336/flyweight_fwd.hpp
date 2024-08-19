

#ifndef BOOST_FLYWEIGHT_FLYWEIGHT_FWD_HPP
#define BOOST_FLYWEIGHT_FLYWEIGHT_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/templated_streams.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/parameter/parameters.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <iosfwd>

#if !defined(BOOST_FLYWEIGHT_DISABLE_HASH_SUPPORT)
#include <boost/functional/hash_fwd.hpp>
#include <cstddef>
#endif

namespace boost{

namespace flyweights{

template<
typename T,
typename Arg1=parameter::void_,
typename Arg2=parameter::void_,
typename Arg3=parameter::void_,
typename Arg4=parameter::void_,
typename Arg5=parameter::void_
>
class flyweight;

#define BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(n)            \
typename Arg##n##1,typename Arg##n##2,typename Arg##n##3, \
typename Arg##n##4,typename Arg##n##5
#define BOOST_FLYWEIGHT_TEMPL_ARGS(n) \
Arg##n##1,Arg##n##2,Arg##n##3,Arg##n##4,Arg##n##5

template<
typename T1,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(1),
typename T2,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(2)
>
bool operator==(
const flyweight<T1,BOOST_FLYWEIGHT_TEMPL_ARGS(1)>& x,
const flyweight<T2,BOOST_FLYWEIGHT_TEMPL_ARGS(2)>& y);

template<
typename T1,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(1),
typename T2,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(2)
>
bool operator<(
const flyweight<T1,BOOST_FLYWEIGHT_TEMPL_ARGS(1)>& x,
const flyweight<T2,BOOST_FLYWEIGHT_TEMPL_ARGS(2)>& y);

#if !defined(BOOST_NO_FUNCTION_TEMPLATE_ORDERING)
template<
typename T1,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(1),
typename T2
>
bool operator==(
const flyweight<T1,BOOST_FLYWEIGHT_TEMPL_ARGS(1)>& x,const T2& y);

template<
typename T1,
typename T2,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(2)
>
bool operator==(
const T1& x,const flyweight<T2,BOOST_FLYWEIGHT_TEMPL_ARGS(2)>& y);

template<
typename T1,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(1),
typename T2
>
bool operator<(
const flyweight<T1,BOOST_FLYWEIGHT_TEMPL_ARGS(1)>& x,const T2& y);

template<
typename T1,
typename T2,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(2)
>
bool operator<(
const T1& x,const flyweight<T2,BOOST_FLYWEIGHT_TEMPL_ARGS(2)>& y);
#endif 

#define BOOST_FLYWEIGHT_COMPLETE_COMP_OPS_DECL(t,a1,a2)                       \
template<t>                                                                   \
inline bool operator!=(const a1& x,const a2& y);                              \
\
template<t>                                                                   \
inline bool operator>(const a1& x,const a2& y);                               \
\
template<t>                                                                   \
inline bool operator>=(const a1& x,const a2& y);                              \
\
template<t>                                                                   \
inline bool operator<=(const a1& x,const a2& y);                              \

BOOST_FLYWEIGHT_COMPLETE_COMP_OPS_DECL(
typename T1 BOOST_PP_COMMA()
BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(1) BOOST_PP_COMMA()
typename T2 BOOST_PP_COMMA()
BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(2),
flyweight<
T1 BOOST_PP_COMMA() BOOST_FLYWEIGHT_TEMPL_ARGS(1)
>,
flyweight<
T2 BOOST_PP_COMMA() BOOST_FLYWEIGHT_TEMPL_ARGS(2)
>)

#if !defined(BOOST_NO_FUNCTION_TEMPLATE_ORDERING)
BOOST_FLYWEIGHT_COMPLETE_COMP_OPS_DECL(
typename T1 BOOST_PP_COMMA()
BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(1) BOOST_PP_COMMA()
typename T2,
flyweight<
T1 BOOST_PP_COMMA() BOOST_FLYWEIGHT_TEMPL_ARGS(1)
>,
T2)

BOOST_FLYWEIGHT_COMPLETE_COMP_OPS_DECL(
typename T1 BOOST_PP_COMMA()
typename T2 BOOST_PP_COMMA()
BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(2),
T1,
flyweight<
T2 BOOST_PP_COMMA() BOOST_FLYWEIGHT_TEMPL_ARGS(2)
>)
#endif 

template<typename T,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(_)>
inline void swap(
flyweight<T,BOOST_FLYWEIGHT_TEMPL_ARGS(_)>& x,
flyweight<T,BOOST_FLYWEIGHT_TEMPL_ARGS(_)>& y);

template<
BOOST_TEMPLATED_STREAM_ARGS(ElemType,Traits)
BOOST_TEMPLATED_STREAM_COMMA
typename T,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(_)
>
inline BOOST_TEMPLATED_STREAM(ostream,ElemType,Traits)& operator<<(
BOOST_TEMPLATED_STREAM(ostream,ElemType,Traits)& out,
const flyweight<T,BOOST_FLYWEIGHT_TEMPL_ARGS(_)>& x);

template<
BOOST_TEMPLATED_STREAM_ARGS(ElemType,Traits)
BOOST_TEMPLATED_STREAM_COMMA
typename T,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(_)
>
inline BOOST_TEMPLATED_STREAM(istream,ElemType,Traits)& operator>>(
BOOST_TEMPLATED_STREAM(istream,ElemType,Traits)& in,
flyweight<T,BOOST_FLYWEIGHT_TEMPL_ARGS(_)>& x);

} 

using flyweights::flyweight;

} 

#if !defined(BOOST_FLYWEIGHT_DISABLE_HASH_SUPPORT)
#if !defined(BOOST_NO_CXX11_HDR_FUNCTIONAL)

#if BOOST_WORKAROUND(_CPPLIB_VER,==520)

#define BOOST_FLYWEIGHT_STD_HASH_STRUCT_KEYWORD class
#else
#define BOOST_FLYWEIGHT_STD_HASH_STRUCT_KEYWORD struct
#endif

#if !defined(_LIBCPP_VERSION)
namespace std{
template <class T> BOOST_FLYWEIGHT_STD_HASH_STRUCT_KEYWORD hash;
}
#else 

#include <functional>
#endif

namespace std{

template<typename T,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(_)>
BOOST_FLYWEIGHT_STD_HASH_STRUCT_KEYWORD
hash<boost::flyweight<T,BOOST_FLYWEIGHT_TEMPL_ARGS(_)> >;

} 
#endif 

namespace boost{
#if !defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
namespace flyweights{
#endif

template<typename T,BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS(_)>
inline std::size_t hash_value(
const flyweight<T,BOOST_FLYWEIGHT_TEMPL_ARGS(_)>& x);

#if !defined(BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP)
} 
#endif
} 
#endif 

#undef BOOST_FLYWEIGHT_COMPLETE_COMP_OPS_DECL
#undef BOOST_FLYWEIGHT_TEMPL_ARGS
#undef BOOST_FLYWEIGHT_TYPENAME_TEMPL_ARGS

#endif
