
#ifndef BOOST_VARIANT_MULTIVISITORS_HPP
#define BOOST_VARIANT_MULTIVISITORS_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>
#include <boost/variant/variant_fwd.hpp> 

#if !defined(BOOST_VARIANT_DO_NOT_USE_VARIADIC_TEMPLATES) && !defined(BOOST_NO_CXX11_HDR_TUPLE)
#   include <boost/variant/detail/multivisitors_cpp11_based.hpp>
#   if !defined(BOOST_NO_CXX14_DECLTYPE_AUTO) && !defined(BOOST_NO_CXX11_DECLTYPE_N3276)
#       include <boost/variant/detail/multivisitors_cpp14_based.hpp>
#   endif
#else
#   include <boost/variant/detail/multivisitors_preprocessor_based.hpp>
#endif

#endif 

