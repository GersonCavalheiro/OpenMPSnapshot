
#ifndef BOOST_ASIO_DETAIL_TIMER_QUEUE_SET_HPP
#define BOOST_ASIO_DETAIL_TIMER_QUEUE_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/timer_queue_base.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class timer_queue_set
{
public:
BOOST_ASIO_DECL timer_queue_set();

BOOST_ASIO_DECL void insert(timer_queue_base* q);

BOOST_ASIO_DECL void erase(timer_queue_base* q);

BOOST_ASIO_DECL bool all_empty() const;

BOOST_ASIO_DECL long wait_duration_msec(long max_duration) const;

BOOST_ASIO_DECL long wait_duration_usec(long max_duration) const;

BOOST_ASIO_DECL void get_ready_timers(op_queue<operation>& ops);

BOOST_ASIO_DECL void get_all_timers(op_queue<operation>& ops);

private:
timer_queue_base* first_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/timer_queue_set.ipp>
#endif 

#endif 
