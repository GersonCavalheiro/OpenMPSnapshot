
#ifndef ASIO_DETAIL_POSIX_EVENT_HPP
#define ASIO_DETAIL_POSIX_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include <cstddef>
#include <pthread.h>
#include "asio/detail/assert.hpp"
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class posix_event
: private noncopyable
{
public:
ASIO_DECL posix_event();

~posix_event()
{
::pthread_cond_destroy(&cond_);
}

template <typename Lock>
void signal(Lock& lock)
{
this->signal_all(lock);
}

template <typename Lock>
void signal_all(Lock& lock)
{
ASIO_ASSERT(lock.locked());
(void)lock;
state_ |= 1;
::pthread_cond_broadcast(&cond_); 
}

template <typename Lock>
void unlock_and_signal_one(Lock& lock)
{
ASIO_ASSERT(lock.locked());
state_ |= 1;
bool have_waiters = (state_ > 1);
lock.unlock();
if (have_waiters)
::pthread_cond_signal(&cond_); 
}

template <typename Lock>
void unlock_and_signal_one_for_destruction(Lock& lock)
{
ASIO_ASSERT(lock.locked());
state_ |= 1;
bool have_waiters = (state_ > 1);
if (have_waiters)
::pthread_cond_signal(&cond_); 
lock.unlock();
}

template <typename Lock>
bool maybe_unlock_and_signal_one(Lock& lock)
{
ASIO_ASSERT(lock.locked());
state_ |= 1;
if (state_ > 1)
{
lock.unlock();
::pthread_cond_signal(&cond_); 
return true;
}
return false;
}

template <typename Lock>
void clear(Lock& lock)
{
ASIO_ASSERT(lock.locked());
(void)lock;
state_ &= ~std::size_t(1);
}

template <typename Lock>
void wait(Lock& lock)
{
ASIO_ASSERT(lock.locked());
while ((state_ & 1) == 0)
{
state_ += 2;
::pthread_cond_wait(&cond_, &lock.mutex().mutex_); 
state_ -= 2;
}
}

template <typename Lock>
bool wait_for_usec(Lock& lock, long usec)
{
ASIO_ASSERT(lock.locked());
if ((state_ & 1) == 0)
{
state_ += 2;
timespec ts;
#if (defined(__MACH__) && defined(__APPLE__)) \
|| (defined(__ANDROID__) && (__ANDROID_API__ < 21) \
&& defined(HAVE_PTHREAD_COND_TIMEDWAIT_RELATIVE))
ts.tv_sec = usec / 1000000;
ts.tv_nsec = (usec % 1000000) * 1000;
::pthread_cond_timedwait_relative_np(
&cond_, &lock.mutex().mutex_, &ts); 
#else 
if (::clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
{
ts.tv_sec += usec / 1000000;
ts.tv_nsec += (usec % 1000000) * 1000;
ts.tv_sec += ts.tv_nsec / 1000000000;
ts.tv_nsec = ts.tv_nsec % 1000000000;
::pthread_cond_timedwait(&cond_,
&lock.mutex().mutex_, &ts); 
}
#endif 
state_ -= 2;
}
return (state_ & 1) != 0;
}

private:
::pthread_cond_t cond_;
std::size_t state_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/posix_event.ipp"
#endif 

#endif 

#endif 
