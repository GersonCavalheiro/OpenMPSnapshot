
#ifndef BOOST_ASIO_DETAIL_WIN_STATIC_MUTEX_HPP
#define BOOST_ASIO_DETAIL_WIN_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS)

#include <boost/asio/detail/scoped_lock.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

struct win_static_mutex
{
typedef boost::asio::detail::scoped_lock<win_static_mutex> scoped_lock;

BOOST_ASIO_DECL void init();

BOOST_ASIO_DECL int do_init();

void lock()
{
::EnterCriticalSection(&crit_section_);
}

void unlock()
{
::LeaveCriticalSection(&crit_section_);
}

bool initialised_;
::CRITICAL_SECTION crit_section_;
};

#if defined(UNDER_CE)
# define BOOST_ASIO_WIN_STATIC_MUTEX_INIT { false, { 0, 0, 0, 0, 0 } }
#else 
# define BOOST_ASIO_WIN_STATIC_MUTEX_INIT { false, { 0, 0, 0, 0, 0, 0 } }
#endif 

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_static_mutex.ipp>
#endif 

#endif 

#endif 
