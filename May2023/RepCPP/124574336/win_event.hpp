
#ifndef BOOST_ASIO_DETAIL_WIN_EVENT_HPP
#define BOOST_ASIO_DETAIL_WIN_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS)

#include <cstddef>
#include <boost/asio/detail/assert.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class win_event
: private noncopyable
{
public:
BOOST_ASIO_DECL win_event();

BOOST_ASIO_DECL ~win_event();

template <typename Lock>
void signal(Lock& lock)
{
this->signal_all(lock);
}

template <typename Lock>
void signal_all(Lock& lock)
{
BOOST_ASIO_ASSERT(lock.locked());
(void)lock;
state_ |= 1;
::SetEvent(events_[0]);
}

template <typename Lock>
void unlock_and_signal_one(Lock& lock)
{
BOOST_ASIO_ASSERT(lock.locked());
state_ |= 1;
bool have_waiters = (state_ > 1);
lock.unlock();
if (have_waiters)
::SetEvent(events_[1]);
}

template <typename Lock>
void unlock_and_signal_one_for_destruction(Lock& lock)
{
BOOST_ASIO_ASSERT(lock.locked());
state_ |= 1;
bool have_waiters = (state_ > 1);
if (have_waiters)
::SetEvent(events_[1]);
lock.unlock();
}

template <typename Lock>
bool maybe_unlock_and_signal_one(Lock& lock)
{
BOOST_ASIO_ASSERT(lock.locked());
state_ |= 1;
if (state_ > 1)
{
lock.unlock();
::SetEvent(events_[1]);
return true;
}
return false;
}

template <typename Lock>
void clear(Lock& lock)
{
BOOST_ASIO_ASSERT(lock.locked());
(void)lock;
::ResetEvent(events_[0]);
state_ &= ~std::size_t(1);
}

template <typename Lock>
void wait(Lock& lock)
{
BOOST_ASIO_ASSERT(lock.locked());
while ((state_ & 1) == 0)
{
state_ += 2;
lock.unlock();
#if defined(BOOST_ASIO_WINDOWS_APP)
::WaitForMultipleObjectsEx(2, events_, false, INFINITE, false);
#else 
::WaitForMultipleObjects(2, events_, false, INFINITE);
#endif 
lock.lock();
state_ -= 2;
}
}

template <typename Lock>
bool wait_for_usec(Lock& lock, long usec)
{
BOOST_ASIO_ASSERT(lock.locked());
if ((state_ & 1) == 0)
{
state_ += 2;
lock.unlock();
DWORD msec = usec > 0 ? (usec < 1000 ? 1 : usec / 1000) : 0;
#if defined(BOOST_ASIO_WINDOWS_APP)
::WaitForMultipleObjectsEx(2, events_, false, msec, false);
#else 
::WaitForMultipleObjects(2, events_, false, msec);
#endif 
lock.lock();
state_ -= 2;
}
return (state_ & 1) != 0;
}

private:
HANDLE events_[2];
std::size_t state_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_event.ipp>
#endif 

#endif 

#endif 
