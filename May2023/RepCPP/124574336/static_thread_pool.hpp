
#ifndef BOOST_ASIO_STATIC_THREAD_POOL_HPP
#define BOOST_ASIO_STATIC_THREAD_POOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/thread_pool.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

typedef thread_pool static_thread_pool;

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
