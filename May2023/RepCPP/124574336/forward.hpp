

#ifndef BOOST_IOSTREAMS_DETAIL_FORWARD_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_FORWARD_HPP_INCLUDED   

#if defined(_MSC_VER)
# pragma once
#endif                  

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/config/limits.hpp>
#include <boost/iostreams/detail/push_params.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/type_traits/is_same.hpp>


#define BOOST_IOSTREAMS_FORWARD(class, impl, device, params, args) \
class(const device& t params()) \
{ this->impl(::boost::iostreams::detail::wrap(t) args()); } \
class(device& t params()) \
{ this->impl(::boost::iostreams::detail::wrap(t) args()); } \
class(const ::boost::reference_wrapper<device>& ref params()) \
{ this->impl(ref args()); } \
void open(const device& t params()) \
{ this->impl(::boost::iostreams::detail::wrap(t) args()); } \
void open(device& t params()) \
{ this->impl(::boost::iostreams::detail::wrap(t) args()); } \
void open(const ::boost::reference_wrapper<device>& ref params()) \
{ this->impl(ref args()); } \
BOOST_PP_REPEAT_FROM_TO( \
1, BOOST_PP_INC(BOOST_IOSTREAMS_MAX_FORWARDING_ARITY), \
BOOST_IOSTREAMS_FORWARDING_CTOR, (class, impl, device) \
) \
BOOST_PP_REPEAT_FROM_TO( \
1, BOOST_PP_INC(BOOST_IOSTREAMS_MAX_FORWARDING_ARITY), \
BOOST_IOSTREAMS_FORWARDING_FN, (class, impl, device) \
) \

#define BOOST_IOSTREAMS_FORWARDING_CTOR(z, n, tuple) \
template<BOOST_PP_ENUM_PARAMS_Z(z, n, typename U)> \
BOOST_PP_TUPLE_ELEM(3, 0, tuple) \
(BOOST_PP_ENUM_BINARY_PARAMS_Z(z, n, const U, &u) \
BOOST_IOSTREAMS_DISABLE_IF_SAME(U0, BOOST_PP_TUPLE_ELEM(3, 2, tuple))) \
{ this->BOOST_PP_TUPLE_ELEM(3, 1, tuple) \
( BOOST_PP_TUPLE_ELEM(3, 2, tuple) \
(BOOST_PP_ENUM_PARAMS_Z(z, n, u)) ); } \
template< typename U100 BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_DEC(n), typename U) > \
BOOST_PP_TUPLE_ELEM(3, 0, tuple) \
( U100& u100 BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_DEC(n), const U, &u) \
BOOST_IOSTREAMS_DISABLE_IF_SAME(U100, BOOST_PP_TUPLE_ELEM(3, 2, tuple))) \
{ this->BOOST_PP_TUPLE_ELEM(3, 1, tuple) \
( BOOST_PP_TUPLE_ELEM(3, 2, tuple) \
( u100 BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_DEC(n), u)) ); } \

#define BOOST_IOSTREAMS_FORWARDING_FN(z, n, tuple) \
template<BOOST_PP_ENUM_PARAMS_Z(z, n, typename U)> \
void open(BOOST_PP_ENUM_BINARY_PARAMS_Z(z, n, const U, &u) \
BOOST_IOSTREAMS_DISABLE_IF_SAME(U0, BOOST_PP_TUPLE_ELEM(3, 2, tuple))) \
{ this->BOOST_PP_TUPLE_ELEM(3, 1, tuple) \
( BOOST_PP_TUPLE_ELEM(3, 2, tuple) \
(BOOST_PP_ENUM_PARAMS_Z(z, n, u)) ); } \
template< typename U100 BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_DEC(n), typename U) > \
void open \
( U100& u100 BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_BINARY_PARAMS_Z(z, BOOST_PP_DEC(n), const U, &u) \
BOOST_IOSTREAMS_DISABLE_IF_SAME(U100, BOOST_PP_TUPLE_ELEM(3, 2, tuple))) \
{ this->BOOST_PP_TUPLE_ELEM(3, 1, tuple) \
( u100 BOOST_PP_COMMA_IF(BOOST_PP_DEC(n)) \
BOOST_PP_ENUM_PARAMS_Z(z, BOOST_PP_DEC(n), u) ); } \


#if !defined(BOOST_NO_SFINAE) && \
!BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x592))
# define BOOST_IOSTREAMS_DISABLE_IF_SAME(device, param) \
, typename boost::disable_if< boost::is_same<device, param> >::type* = 0 \

#else 
# define BOOST_IOSTREAMS_DISABLE_IF_SAME(device, param)
#endif

#endif 
