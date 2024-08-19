
#ifndef ASIO_DETAIL_STD_EVENT_HPP
#define ASIO_DETAIL_STD_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_STD_MUTEX_AND_CONDVAR)

#include <chrono>
#include <condition_variable>
#include "asio/detail/assert.hpp"
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class std_event
: private noncopyable
{
public:
std_event()
: state_(0)
{
}

~std_event()
{
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
cond_.notify_all();
}

template <typename Lock>
void unlock_and_signal_one(Lock& lock)
{
ASIO_ASSERT(lock.locked());
state_ |= 1;
bool have_waiters = (state_ > 1);
lock.unlock();
if (have_waiters)
cond_.notify_one();
}

template <typename Lock>
void unlock_and_signal_one_for_destruction(Lock& lock)
{
ASIO_ASSERT(lock.locked());
state_ |= 1;
bool have_waiters = (state_ > 1);
if (have_waiters)
cond_.notify_one();
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
cond_.notify_one();
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
unique_lock_adapter u_lock(lock);
while ((state_ & 1) == 0)
{
waiter w(state_);
cond_.wait(u_lock.unique_lock_);
}
}

template <typename Lock>
bool wait_for_usec(Lock& lock, long usec)
{
ASIO_ASSERT(lock.locked());
unique_lock_adapter u_lock(lock);
if ((state_ & 1) == 0)
{
waiter w(state_);
cond_.wait_for(u_lock.unique_lock_, std::chrono::microseconds(usec));
}
return (state_ & 1) != 0;
}

private:
struct unique_lock_adapter
{
template <typename Lock>
explicit unique_lock_adapter(Lock& lock)
: unique_lock_(lock.mutex().mutex_, std::adopt_lock)
{
}

~unique_lock_adapter()
{
unique_lock_.release();
}

std::unique_lock<std::mutex> unique_lock_;
};

class waiter
{
public:
explicit waiter(std::size_t& state)
: state_(state)
{
state_ += 2;
}

~waiter()
{
state_ -= 2;
}

private:
std::size_t& state_;
};

std::condition_variable cond_;
std::size_t state_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
