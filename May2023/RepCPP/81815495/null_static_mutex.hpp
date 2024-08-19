
#ifndef ASIO_DETAIL_NULL_STATIC_MUTEX_HPP
#define ASIO_DETAIL_NULL_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)

#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct null_static_mutex
{
typedef asio::detail::scoped_lock<null_static_mutex> scoped_lock;

void init()
{
}

void lock()
{
}

void unlock()
{
}

int unused_;
};

#define ASIO_NULL_STATIC_MUTEX_INIT { 0 }

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
