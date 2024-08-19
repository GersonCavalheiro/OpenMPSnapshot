
#ifndef ASIO_STATIC_THREAD_POOL_HPP
#define ASIO_STATIC_THREAD_POOL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/thread_pool.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

typedef thread_pool static_thread_pool;

} 

#include "asio/detail/pop_options.hpp"

#endif 
