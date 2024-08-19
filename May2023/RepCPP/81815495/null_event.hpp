
#ifndef ASIO_DETAIL_NULL_EVENT_HPP
#define ASIO_DETAIL_NULL_EVENT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class null_event
: private noncopyable
{
public:
null_event()
{
}

~null_event()
{
}

template <typename Lock>
void signal(Lock&)
{
}

template <typename Lock>
void signal_all(Lock&)
{
}

template <typename Lock>
void unlock_and_signal_one(Lock&)
{
}

template <typename Lock>
void unlock_and_signal_one_for_destruction(Lock&)
{
}

template <typename Lock>
bool maybe_unlock_and_signal_one(Lock&)
{
return false;
}

template <typename Lock>
void clear(Lock&)
{
}

template <typename Lock>
void wait(Lock&)
{
do_wait();
}

template <typename Lock>
bool wait_for_usec(Lock&, long usec)
{
do_wait_for_usec(usec);
return true;
}

private:
ASIO_DECL static void do_wait();
ASIO_DECL static void do_wait_for_usec(long usec);
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/null_event.ipp"
#endif 

#endif 
