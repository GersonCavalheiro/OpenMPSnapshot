
#ifndef BOOST_ASIO_DETAIL_WINRT_SOCKET_CONNECT_OP_HPP
#define BOOST_ASIO_DETAIL_WINRT_SOCKET_CONNECT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)

#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/winrt_async_op.hpp>
#include <boost/asio/error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Handler, typename IoExecutor>
class winrt_socket_connect_op :
public winrt_async_op<void>
{
public:
BOOST_ASIO_DEFINE_HANDLER_PTR(winrt_socket_connect_op);

winrt_socket_connect_op(Handler& handler, const IoExecutor& io_ex)
: winrt_async_op<void>(&winrt_socket_connect_op::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code&, std::size_t)
{
winrt_socket_connect_op* o(static_cast<winrt_socket_connect_op*>(base));
ptr p = { boost::asio::detail::addressof(o->handler_), o, o };

BOOST_ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder1<Handler, boost::system::error_code>
handler(o->handler_, o->ec_);
p.h = boost::asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_));
w.complete(handler, handler.handler_);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
handler_work<Handler, IoExecutor> executor_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
