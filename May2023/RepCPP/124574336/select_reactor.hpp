
#ifndef BOOST_ASIO_DETAIL_IMPL_SELECT_REACTOR_HPP
#define BOOST_ASIO_DETAIL_IMPL_SELECT_REACTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP) \
|| (!defined(BOOST_ASIO_HAS_DEV_POLL) \
&& !defined(BOOST_ASIO_HAS_EPOLL) \
&& !defined(BOOST_ASIO_HAS_KQUEUE) \
&& !defined(BOOST_ASIO_WINDOWS_RUNTIME))

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Time_Traits>
void select_reactor::add_timer_queue(timer_queue<Time_Traits>& queue)
{
do_add_timer_queue(queue);
}

template <typename Time_Traits>
void select_reactor::remove_timer_queue(timer_queue<Time_Traits>& queue)
{
do_remove_timer_queue(queue);
}

template <typename Time_Traits>
void select_reactor::schedule_timer(timer_queue<Time_Traits>& queue,
const typename Time_Traits::time_type& time,
typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op)
{
boost::asio::detail::mutex::scoped_lock lock(mutex_);

if (shutdown_)
{
scheduler_.post_immediate_completion(op, false);
return;
}

bool earliest = queue.enqueue_timer(time, timer, op);
scheduler_.work_started();
if (earliest)
interrupter_.interrupt();
}

template <typename Time_Traits>
std::size_t select_reactor::cancel_timer(timer_queue<Time_Traits>& queue,
typename timer_queue<Time_Traits>::per_timer_data& timer,
std::size_t max_cancelled)
{
boost::asio::detail::mutex::scoped_lock lock(mutex_);
op_queue<operation> ops;
std::size_t n = queue.cancel_timer(timer, ops, max_cancelled);
lock.unlock();
scheduler_.post_deferred_completions(ops);
return n;
}

template <typename Time_Traits>
void select_reactor::move_timer(timer_queue<Time_Traits>& queue,
typename timer_queue<Time_Traits>::per_timer_data& target,
typename timer_queue<Time_Traits>::per_timer_data& source)
{
boost::asio::detail::mutex::scoped_lock lock(mutex_);
op_queue<operation> ops;
queue.cancel_timer(target, ops);
queue.move_timer(target, source);
lock.unlock();
scheduler_.post_deferred_completions(ops);
}

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
