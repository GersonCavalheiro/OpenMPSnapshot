
#ifndef ASIO_DETAIL_WIN_IOCP_HANDLE_READ_OP_HPP
#define ASIO_DETAIL_WIN_IOCP_HANDLE_READ_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/handler_work.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/operation.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename MutableBufferSequence, typename Handler, typename IoExecutor>
class win_iocp_handle_read_op : public operation
{
public:
ASIO_DEFINE_HANDLER_PTR(win_iocp_handle_read_op);

win_iocp_handle_read_op(const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
: operation(&win_iocp_handle_read_op::do_complete),
buffers_(buffers),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t bytes_transferred)
{
asio::error_code ec(result_ec);

win_iocp_handle_read_op* o(static_cast<win_iocp_handle_read_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

#if defined(ASIO_ENABLE_BUFFER_DEBUGGING)
if (owner)
{
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::validate(o->buffers_);
}
#endif 

if (ec.value() == ERROR_HANDLE_EOF)
ec = asio::error::eof;

detail::binder2<Handler, asio::error_code, std::size_t>
handler(o->handler_, ec, bytes_transferred);
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
MutableBufferSequence buffers_;
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 