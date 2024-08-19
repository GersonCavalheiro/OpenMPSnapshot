
#ifndef ASIO_DETAIL_SCOPED_LOCK_HPP
#define ASIO_DETAIL_SCOPED_LOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Mutex>
class scoped_lock
: private noncopyable
{
public:
enum adopt_lock_t { adopt_lock };

scoped_lock(Mutex& m, adopt_lock_t)
: mutex_(m),
locked_(true)
{
}

explicit scoped_lock(Mutex& m)
: mutex_(m)
{
mutex_.lock();
locked_ = true;
}

~scoped_lock()
{
if (locked_)
mutex_.unlock();
}

void lock()
{
if (!locked_)
{
mutex_.lock();
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

Mutex& mutex()
{
return mutex_;
}

private:
Mutex& mutex_;

bool locked_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
