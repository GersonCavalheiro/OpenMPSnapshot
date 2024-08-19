
#ifndef BOOST_ASIO_DETAIL_FUTURE_HPP
#define BOOST_ASIO_DETAIL_FUTURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#if defined(BOOST_ASIO_HAS_STD_FUTURE)
# include <future>
# if defined(__GNUC__) && !defined(BOOST_ASIO_HAS_CLANG_LIBCXX)
#  if defined(_GLIBCXX_HAS_GTHREADS)
#   define BOOST_ASIO_HAS_STD_FUTURE_CLASS 1
#  endif 
# else 
#  define BOOST_ASIO_HAS_STD_FUTURE_CLASS 1
# endif 
#endif 

#endif 
