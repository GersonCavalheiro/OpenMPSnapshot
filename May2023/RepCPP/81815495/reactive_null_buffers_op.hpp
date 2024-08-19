
#ifndef ASIO_DETAIL_REACTIVE_NULL_BUFFERS_OP_HPP
#define ASIO_DETAIL_REACTIVE_NULL_BUFFERS_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_work.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/reactor_op.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler, typename IoExecutor>
class reactive_null_buffers_op : public reactor_op
{
public:
ASIO_DEFINE_HANDLER_PTR(reactive_null_buffers_op);

reactive_null_buffers_op(const asio::error_code& success_ec,
Handler& handler, const IoExecutor& io_ex)
: reactor_op(success_ec, &reactive_null_buffers_op::do_perform,
&reactive_null_buffers_op::do_complete),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static status do_perform(reactor_op*)
{
return done;
}

static void do_complete(void* owner, operation* base,
const asio::error_code& ,
std::size_t )
{
reactive_null_buffers_op* o(static_cast<reactive_null_buffers_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder2<Handler, asio::error_code, std::size_t>
handler(o->handler_, o->ec_, o->bytes_transferred_);
p.h = asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, handler.arg2_));
w.complete(handler, handler.handler_);
ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
