
#ifndef BOOST_ASIO_DETAIL_NULL_STATIC_MUTEX_HPP
#define BOOST_ASIO_DETAIL_NULL_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_THREADS)

#include <boost/asio/detail/scoped_lock.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

struct null_static_mutex
{
typedef boost::asio::detail::scoped_lock<null_static_mutex> scoped_lock;

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

#define BOOST_ASIO_NULL_STATIC_MUTEX_INIT { 0 }

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
