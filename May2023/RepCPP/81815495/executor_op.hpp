
#ifndef ASIO_DETAIL_EXECUTOR_OP_HPP
#define ASIO_DETAIL_EXECUTOR_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/scheduler_operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler, typename Alloc,
typename Operation = scheduler_operation>
class executor_op : public Operation
{
public:
ASIO_DEFINE_HANDLER_ALLOCATOR_PTR(executor_op);

template <typename H>
executor_op(ASIO_MOVE_ARG(H) h, const Alloc& allocator)
: Operation(&executor_op::do_complete),
handler_(ASIO_MOVE_CAST(H)(h)),
allocator_(allocator)
{
}

static void do_complete(void* owner, Operation* base,
const asio::error_code& ,
std::size_t )
{
executor_op* o(static_cast<executor_op*>(base));
Alloc allocator(o->allocator_);
ptr p = { detail::addressof(allocator), o, o };

ASIO_HANDLER_COMPLETION((*o));

Handler handler(ASIO_MOVE_CAST(Handler)(o->handler_));
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
ASIO_HANDLER_INVOCATION_BEGIN(());
asio_handler_invoke_helpers::invoke(handler, handler);
ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
Alloc allocator_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
