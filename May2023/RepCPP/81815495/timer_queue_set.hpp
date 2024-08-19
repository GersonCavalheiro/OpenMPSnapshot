
#ifndef ASIO_DETAIL_TIMER_QUEUE_SET_HPP
#define ASIO_DETAIL_TIMER_QUEUE_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/timer_queue_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class timer_queue_set
{
public:
ASIO_DECL timer_queue_set();

ASIO_DECL void insert(timer_queue_base* q);

ASIO_DECL void erase(timer_queue_base* q);

ASIO_DECL bool all_empty() const;

ASIO_DECL long wait_duration_msec(long max_duration) const;

ASIO_DECL long wait_duration_usec(long max_duration) const;

ASIO_DECL void get_ready_timers(op_queue<operation>& ops);

ASIO_DECL void get_all_timers(op_queue<operation>& ops);

private:
timer_queue_base* first_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/timer_queue_set.ipp"
#endif 

#endif 
