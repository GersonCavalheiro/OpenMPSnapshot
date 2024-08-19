
#ifndef BOOST_ASIO_DETAIL_TIMER_QUEUE_BASE_HPP
#define BOOST_ASIO_DETAIL_TIMER_QUEUE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/op_queue.hpp>
#include <boost/asio/detail/operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class timer_queue_base
: private noncopyable
{
public:
timer_queue_base() : next_(0) {}

virtual ~timer_queue_base() {}

virtual bool empty() const = 0;

virtual long wait_duration_msec(long max_duration) const = 0;

virtual long wait_duration_usec(long max_duration) const = 0;

virtual void get_ready_timers(op_queue<operation>& ops) = 0;

virtual void get_all_timers(op_queue<operation>& ops) = 0;

private:
friend class timer_queue_set;

timer_queue_base* next_;
};

template <typename Time_Traits>
class timer_queue;

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
