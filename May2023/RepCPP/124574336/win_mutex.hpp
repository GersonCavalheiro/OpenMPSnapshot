
#ifndef BOOST_ASIO_DETAIL_WIN_MUTEX_HPP
#define BOOST_ASIO_DETAIL_WIN_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS)

#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/scoped_lock.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class win_mutex
: private noncopyable
{
public:
typedef boost::asio::detail::scoped_lock<win_mutex> scoped_lock;

BOOST_ASIO_DECL win_mutex();

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
BOOST_ASIO_DECL int do_init();

::CRITICAL_SECTION crit_section_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_mutex.ipp>
#endif 

#endif 

#endif 
