
#ifndef BOOST_ASIO_DETAIL_GLOBAL_HPP
#define BOOST_ASIO_DETAIL_GLOBAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_THREADS)
# include <boost/asio/detail/null_global.hpp>
#elif defined(BOOST_ASIO_WINDOWS)
# include <boost/asio/detail/win_global.hpp>
#elif defined(BOOST_ASIO_HAS_PTHREADS)
# include <boost/asio/detail/posix_global.hpp>
#elif defined(BOOST_ASIO_HAS_STD_CALL_ONCE)
# include <boost/asio/detail/std_global.hpp>
#else
# error Only Windows, POSIX and std::call_once are supported!
#endif

namespace boost {
namespace asio {
namespace detail {

template <typename T>
inline T& global()
{
#if !defined(BOOST_ASIO_HAS_THREADS)
return null_global<T>();
#elif defined(BOOST_ASIO_WINDOWS)
return win_global<T>();
#elif defined(BOOST_ASIO_HAS_PTHREADS)
return posix_global<T>();
#elif defined(BOOST_ASIO_HAS_STD_CALL_ONCE)
return std_global<T>();
#endif
}

} 
} 
} 

#endif 
