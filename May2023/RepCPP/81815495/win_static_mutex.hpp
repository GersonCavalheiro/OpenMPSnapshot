
#ifndef ASIO_DETAIL_WIN_STATIC_MUTEX_HPP
#define ASIO_DETAIL_WIN_STATIC_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS)

#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

struct win_static_mutex
{
typedef asio::detail::scoped_lock<win_static_mutex> scoped_lock;

ASIO_DECL void init();

ASIO_DECL int do_init();

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
# define ASIO_WIN_STATIC_MUTEX_INIT { false, { 0, 0, 0, 0, 0 } }
#else 
# define ASIO_WIN_STATIC_MUTEX_INIT { false, { 0, 0, 0, 0, 0, 0 } }
#endif 

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_static_mutex.ipp"
#endif 

#endif 

#endif 
