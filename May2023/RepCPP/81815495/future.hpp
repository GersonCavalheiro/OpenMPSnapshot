
#ifndef ASIO_DETAIL_FUTURE_HPP
#define ASIO_DETAIL_FUTURE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#if defined(ASIO_HAS_STD_FUTURE)
# include <future>
# if defined(__GNUC__) && !defined(ASIO_HAS_CLANG_LIBCXX)
#  if defined(_GLIBCXX_HAS_GTHREADS)
#   define ASIO_HAS_STD_FUTURE_CLASS 1
#  endif 
# else 
#  define ASIO_HAS_STD_FUTURE_CLASS 1
# endif 
#endif 

#endif 
