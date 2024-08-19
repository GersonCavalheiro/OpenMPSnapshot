
#ifndef ASIO_DETAIL_IMPL_WIN_IOCP_IO_CONTEXT_HPP
#define ASIO_DETAIL_IMPL_WIN_IOCP_IO_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/completion_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/memory.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Time_Traits>
void win_iocp_io_context::add_timer_queue(
timer_queue<Time_Traits>& queue)
{
do_add_timer_queue(queue);
}

template <typename Time_Traits>
void win_iocp_io_context::remove_timer_queue(
timer_queue<Time_Traits>& queue)
{
do_remove_timer_queue(queue);
}

template <typename Time_Traits>
void win_iocp_io_context::schedule_timer(timer_queue<Time_Traits>& queue,
const typename Time_Traits::time_type& time,
typename timer_queue<Time_Traits>::per_timer_data& timer, wait_op* op)
{
if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
{
post_immediate_completion(op, false);
return;
}

mutex::scoped_lock lock(dispatch_mutex_);

bool earliest = queue.enqueue_timer(time, timer, op);
work_started();
if (earliest)
update_timeout();
}

template <typename Time_Traits>
std::size_t win_iocp_io_context::cancel_timer(timer_queue<Time_Traits>& queue,
typename timer_queue<Time_Traits>::per_timer_data& timer,
std::size_t max_cancelled)
{
if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
return 0;

mutex::scoped_lock lock(dispatch_mutex_);
op_queue<win_iocp_operation> ops;
std::size_t n = queue.cancel_timer(timer, ops, max_cancelled);
lock.unlock();
post_deferred_completions(ops);
return n;
}

template <typename Time_Traits>
void win_iocp_io_context::cancel_timer_by_key(timer_queue<Time_Traits>& queue,
typename timer_queue<Time_Traits>::per_timer_data* timer,
void* cancellation_key)
{
if (::InterlockedExchangeAdd(&shutdown_, 0) != 0)
return;

mutex::scoped_lock lock(dispatch_mutex_);
op_queue<win_iocp_operation> ops;
queue.cancel_timer_by_key(timer, ops, cancellation_key);
lock.unlock();
post_deferred_completions(ops);
}

template <typename Time_Traits>
void win_iocp_io_context::move_timer(timer_queue<Time_Traits>& queue,
typename timer_queue<Time_Traits>::per_timer_data& to,
typename timer_queue<Time_Traits>::per_timer_data& from)
{
asio::detail::mutex::scoped_lock lock(dispatch_mutex_);
op_queue<operation> ops;
queue.cancel_timer(to, ops);
queue.move_timer(to, from);
lock.unlock();
post_deferred_completions(ops);
}

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 