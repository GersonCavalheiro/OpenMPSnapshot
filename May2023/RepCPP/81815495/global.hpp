
#ifndef ASIO_DETAIL_GLOBAL_HPP
#define ASIO_DETAIL_GLOBAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)
# include "asio/detail/null_global.hpp"
#elif defined(ASIO_WINDOWS)
# include "asio/detail/win_global.hpp"
#elif defined(ASIO_HAS_PTHREADS)
# include "asio/detail/posix_global.hpp"
#elif defined(ASIO_HAS_STD_CALL_ONCE)
# include "asio/detail/std_global.hpp"
#else
# error Only Windows, POSIX and std::call_once are supported!
#endif

namespace asio {
namespace detail {

template <typename T>
inline T& global()
{
#if !defined(ASIO_HAS_THREADS)
return null_global<T>();
#elif defined(ASIO_WINDOWS)
return win_global<T>();
#elif defined(ASIO_HAS_PTHREADS)
return posix_global<T>();
#elif defined(ASIO_HAS_STD_CALL_ONCE)
return std_global<T>();
#endif
}

} 
} 

#endif 
