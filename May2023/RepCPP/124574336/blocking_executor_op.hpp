
#ifndef BOOST_ASIO_DETAIL_BLOCKING_EXECUTOR_OP_HPP
#define BOOST_ASIO_DETAIL_BLOCKING_EXECUTOR_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/event.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/mutex.hpp>
#include <boost/asio/detail/scheduler_operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Operation = scheduler_operation>
class blocking_executor_op_base : public Operation
{
public:
blocking_executor_op_base(typename Operation::func_type complete_func)
: Operation(complete_func),
is_complete_(false)
{
}

void wait()
{
boost::asio::detail::mutex::scoped_lock lock(mutex_);
while (!is_complete_)
event_.wait(lock);
}

protected:
struct do_complete_cleanup
{
~do_complete_cleanup()
{
boost::asio::detail::mutex::scoped_lock lock(op_->mutex_);
op_->is_complete_ = true;
op_->event_.unlock_and_signal_one_for_destruction(lock);
}

blocking_executor_op_base* op_;
};

private:
boost::asio::detail::mutex mutex_;
boost::asio::detail::event event_;
bool is_complete_;
};

template <typename Handler, typename Operation = scheduler_operation>
class blocking_executor_op : public blocking_executor_op_base<Operation>
{
public:
blocking_executor_op(Handler& h)
: blocking_executor_op_base<Operation>(&blocking_executor_op::do_complete),
handler_(h)
{
}

static void do_complete(void* owner, Operation* base,
const boost::system::error_code& ,
std::size_t )
{
blocking_executor_op* o(static_cast<blocking_executor_op*>(base));

typename blocking_executor_op_base<Operation>::do_complete_cleanup
on_exit = { o };
(void)on_exit;

BOOST_ASIO_HANDLER_COMPLETION((*o));

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN(());
boost_asio_handler_invoke_helpers::invoke(o->handler_, o->handler_);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler& handler_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
