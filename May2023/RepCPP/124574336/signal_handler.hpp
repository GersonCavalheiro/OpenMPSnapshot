
#ifndef BOOST_ASIO_DETAIL_SIGNAL_HANDLER_HPP
#define BOOST_ASIO_DETAIL_SIGNAL_HANDLER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/signal_op.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Handler, typename IoExecutor>
class signal_handler : public signal_op
{
public:
BOOST_ASIO_DEFINE_HANDLER_PTR(signal_handler);

signal_handler(Handler& h, const IoExecutor& io_ex)
: signal_op(&signal_handler::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(h)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code& ,
std::size_t )
{
signal_handler* h(static_cast<signal_handler*>(base));
ptr p = { boost::asio::detail::addressof(h->handler_), h, h };

BOOST_ASIO_HANDLER_COMPLETION((*h));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
h->work_));

detail::binder2<Handler, boost::system::error_code, int>
handler(h->handler_, h->ec_, h->signal_number_);
p.h = boost::asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, handler.arg2_));
w.complete(handler, handler.handler_);
BOOST_ASIO_HANDLER_INVOCATION_END;
}
}

private:
Handler handler_;
handler_work<Handler, IoExecutor> work_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
