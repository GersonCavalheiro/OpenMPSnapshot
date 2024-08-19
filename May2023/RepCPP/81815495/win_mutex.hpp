
#ifndef ASIO_DETAIL_WIN_MUTEX_HPP
#define ASIO_DETAIL_WIN_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/scoped_lock.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_mutex
: private noncopyable
{
public:
typedef asio::detail::scoped_lock<win_mutex> scoped_lock;

ASIO_DECL win_mutex();

~win_mutex()
{
::DeleteCriticalSection(&crit_section_);
}

void lock()
{
::EnterCriticalSection(&crit_section_);
}

void unlock()
{
::LeaveCriticalSection(&crit_section_);
}

private:
ASIO_DECL int do_init();

::CRITICAL_SECTION crit_section_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_mutex.ipp"
#endif 

#endif 

#endif 
