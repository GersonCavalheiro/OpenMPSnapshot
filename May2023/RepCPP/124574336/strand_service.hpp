
#ifndef BOOST_ASIO_DETAIL_IMPL_STRAND_SERVICE_HPP
#define BOOST_ASIO_DETAIL_IMPL_STRAND_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/call_stack.hpp>
#include <boost/asio/detail/completion_handler.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/memory.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

inline strand_service::strand_impl::strand_impl()
: operation(&strand_service::do_complete),
locked_(false)
{
}

struct strand_service::on_dispatch_exit
{
io_context_impl* io_context_impl_;
strand_impl* impl_;

~on_dispatch_exit()
{
impl_->mutex_.lock();
impl_->ready_queue_.push(impl_->waiting_queue_);
bool more_handlers = impl_->locked_ = !impl_->ready_queue_.empty();
impl_->mutex_.unlock();

if (more_handlers)
io_context_impl_->post_immediate_completion(impl_, false);
}
};

template <typename Handler>
void strand_service::dispatch(strand_service::implementation_type& impl,
Handler& handler)
{
if (call_stack<strand_impl>::contains(impl))
{
fenced_block b(fenced_block::full);
boost_asio_handler_invoke_helpers::invoke(handler, handler);
return;
}

typedef completion_handler<Handler, io_context::executor_type> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(handler, io_context_.get_executor());

BOOST_ASIO_HANDLER_CREATION((this->context(),
*p.p, "strand", impl, 0, "dispatch"));

bool dispatch_immediately = do_dispatch(impl, p.p);
operation* o = p.p;
p.v = p.p = 0;

if (dispatch_immediately)
{
call_stack<strand_impl>::context ctx(impl);

on_dispatch_exit on_exit = { &io_context_impl_, impl };
(void)on_exit;

op::do_complete(&io_context_impl_, o, boost::system::error_code(), 0);
}
}

template <typename Handler>
void strand_service::post(strand_service::implementation_type& impl,
Handler& handler)
{
bool is_continuation =
boost_asio_handler_cont_helpers::is_continuation(handler);

typedef completion_handler<Handler, io_context::executor_type> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(handler, io_context_.get_executor());

BOOST_ASIO_HANDLER_CREATION((this->context(),
*p.p, "strand", impl, 0, "post"));

do_post(impl, p.p, is_continuation);
p.v = p.p = 0;
}

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
