
#ifndef BOOST_ASIO_DETAIL_BULK_EXECUTOR_OP_HPP
#define BOOST_ASIO_DETAIL_BULK_EXECUTOR_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/scheduler_operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Handler, typename Alloc,
typename Operation = scheduler_operation>
class bulk_executor_op : public Operation
{
public:
BOOST_ASIO_DEFINE_HANDLER_ALLOCATOR_PTR(bulk_executor_op);

template <typename H>
bulk_executor_op(BOOST_ASIO_MOVE_ARG(H) h,
const Alloc& allocator, std::size_t i)
: Operation(&bulk_executor_op::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(H)(h)),
allocator_(allocator),
index_(i)
{
}

static void do_complete(void* owner, Operation* base,
const boost::system::error_code& ,
std::size_t )
{
bulk_executor_op* o(static_cast<bulk_executor_op*>(base));
Alloc allocator(o->allocator_);
ptr p = { detail::addressof(allocator), o, o };

BOOST_ASIO_HANDLER_COMPLETION((*o));

detail::binder1<Handler, std::size_t> handler(o->handler_, o->index_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN(());
boost_asio_handler_invoke_helpers::invoke(handler, handler.handler_);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
Alloc allocator_;
std::size_t index_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
