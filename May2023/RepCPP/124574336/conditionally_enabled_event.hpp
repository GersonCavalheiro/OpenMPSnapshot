
#ifndef BOOST_ASIO_DETAIL_CONDITIONALLY_ENABLED_EVENT_HPP
#define BOOST_ASIO_DETAIL_CONDITIONALLY_ENABLED_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/conditionally_enabled_mutex.hpp>
#include <boost/asio/detail/event.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/null_event.hpp>
#include <boost/asio/detail/scoped_lock.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class conditionally_enabled_event
: private noncopyable
{
public:
conditionally_enabled_event()
{
}

~conditionally_enabled_event()
{
}

void signal(conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
event_.signal(lock);
}

void signal_all(conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
event_.signal_all(lock);
}

void unlock_and_signal_one(
conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
event_.unlock_and_signal_one(lock);
}

void unlock_and_signal_one_for_destruction(
conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
event_.unlock_and_signal_one(lock);
}

bool maybe_unlock_and_signal_one(
conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
return event_.maybe_unlock_and_signal_one(lock);
else
return false;
}

void clear(conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
event_.clear(lock);
}

void wait(conditionally_enabled_mutex::scoped_lock& lock)
{
if (lock.mutex_.enabled_)
event_.wait(lock);
else
null_event().wait(lock);
}

bool wait_for_usec(
conditionally_enabled_mutex::scoped_lock& lock, long usec)
{
if (lock.mutex_.enabled_)
return event_.wait_for_usec(lock, usec);
else
return null_event().wait_for_usec(lock, usec);
}

private:
boost::asio::detail::event event_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
