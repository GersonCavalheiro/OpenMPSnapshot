
#ifndef BOOST_ASIO_DETAIL_EXECUTOR_OP_HPP
#define BOOST_ASIO_DETAIL_EXECUTOR_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
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
class executor_op : public Operation
{
public:
BOOST_ASIO_DEFINE_HANDLER_ALLOCATOR_PTR(executor_op);

template <typename H>
executor_op(BOOST_ASIO_MOVE_ARG(H) h, const Alloc& allocator)
: Operation(&executor_op::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(H)(h)),
allocator_(allocator)
{
}

static void do_complete(void* owner, Operation* base,
const boost::system::error_code& ,
std::size_t )
{
executor_op* o(static_cast<executor_op*>(base));
Alloc allocator(o->allocator_);
ptr p = { detail::addressof(allocator), o, o };

BOOST_ASIO_HANDLER_COMPLETION((*o));

Handler handler(BOOST_ASIO_MOVE_CAST(Handler)(o->handler_));
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN(());
boost_asio_handler_invoke_helpers::invoke(handler, handler);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
Alloc allocator_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
