

#ifndef BOOST_FLYWEIGHT_DETAIL_PERFECT_FWD_HPP
#define BOOST_FLYWEIGHT_DETAIL_PERFECT_FWD_HPP

#if defined(_MSC_VER)
#pragma once
#endif



#include <boost/config.hpp> 
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/seq/seq.hpp>

#if !defined(BOOST_NO_CXX11_RVALUE_REFERENCES)
#include <utility>
#endif

#define BOOST_FLYWEIGHT_FORWARD_FORWARD_AUX(z,n,_) \
std::forward<BOOST_PP_CAT(T,n)>(BOOST_PP_CAT(t,n))

#define BOOST_FLYWEIGHT_FORWARD_FORWARD(n) \
BOOST_PP_ENUM(n,BOOST_FLYWEIGHT_FORWARD_FORWARD_AUX,~)

#define BOOST_FLYWEIGHT_FORWARD_ENUM(n) BOOST_PP_ENUM_PARAMS(n,t)

#define BOOST_FLYWEIGHT_FORWARD_PASS(arg) arg

#define BOOST_FLYWEIGHT_FORWARD(args)\
BOOST_PP_CAT(BOOST_FLYWEIGHT_FORWARD_,BOOST_PP_SEQ_HEAD(args))( \
BOOST_PP_SEQ_HEAD(BOOST_PP_SEQ_TAIL(args)))

#if defined(BOOST_NO_CXX11_RVALUE_REFERENCES)||\
defined(BOOST_NO_CXX11_VARIADIC_TEMPLATES)

#if !defined(BOOST_FLYWEIGHT_LIMIT_PERFECT_FWD_ARGS)
#define BOOST_FLYWEIGHT_LIMIT_PERFECT_FWD_ARGS 5
#endif

#if BOOST_FLYWEIGHT_LIMIT_PERFECT_FWD_ARGS<0
#error BOOST_FLYWEIGHT_LIMIT_PERFECT_FWD_ARGS must be >=0
#endif

#if BOOST_FLYWEIGHT_LIMIT_PERFECT_FWD_ARGS<=5
#include <boost/flyweight/detail/pp_perfect_fwd.hpp>
#else
#include <boost/flyweight/detail/dyn_perfect_fwd.hpp>
#endif

#else



#define BOOST_FLYWEIGHT_PERFECT_FWD(name,body) \
template<typename... Args>name(Args&&... args) \
body((PASS)(std::forward<Args>(args)...))

#define BOOST_FLYWEIGHT_PERFECT_FWD_WITH_ARGS  \
BOOST_FLYWEIGHT_PERFECT_FWD

#endif
#endif
