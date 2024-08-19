
#ifndef ASIO_DETAIL_CONDITIONALLY_ENABLED_MUTEX_HPP
#define ASIO_DETAIL_CONDITIONALLY_ENABLED_MUTEX_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/scoped_lock.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class conditionally_enabled_mutex
: private noncopyable
{
public:
class scoped_lock
: private noncopyable
{
public:
enum adopt_lock_t { adopt_lock };

scoped_lock(conditionally_enabled_mutex& m, adopt_lock_t)
: mutex_(m),
locked_(m.enabled_)
{
}

explicit scoped_lock(conditionally_enabled_mutex& m)
: mutex_(m)
{
if (m.enabled_)
{
mutex_.mutex_.lock();
locked_ = true;
}
else
locked_ = false;
}

~scoped_lock()
{
if (locked_)
mutex_.mutex_.unlock();
}

void lock()
{
if (mutex_.enabled_ && !locked_)
{
mutex_.mutex_.lock();
locked_ = true;
}
}

void unlock()
{
if (locked_)
{
mutex_.unlock();
locked_ = false;
}
}

bool locked() const
{
return locked_;
}

asio::detail::mutex& mutex()
{
return mutex_.mutex_;
}

private:
friend class conditionally_enabled_event;
conditionally_enabled_mutex& mutex_;
bool locked_;
};

explicit conditionally_enabled_mutex(bool enabled)
: enabled_(enabled)
{
}

~conditionally_enabled_mutex()
{
}

bool enabled() const
{
return enabled_;
}

void lock()
{
if (enabled_)
mutex_.lock();
}

void unlock()
{
if (enabled_)
mutex_.unlock();
}

private:
friend class scoped_lock;
friend class conditionally_enabled_event;
asio::detail::mutex mutex_;
const bool enabled_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
