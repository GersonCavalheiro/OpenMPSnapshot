
#ifndef BOOST_ASIO_DETAIL_COMPLETION_HANDLER_HPP
#define BOOST_ASIO_DETAIL_COMPLETION_HANDLER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/fenced_block.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_work.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Handler, typename IoExecutor>
class completion_handler : public operation
{
public:
BOOST_ASIO_DEFINE_HANDLER_PTR(completion_handler);

completion_handler(Handler& h, const IoExecutor& io_ex)
: operation(&completion_handler::do_complete),
handler_(BOOST_ASIO_MOVE_CAST(Handler)(h)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const boost::system::error_code& ,
std::size_t )
{
completion_handler* h(static_cast<completion_handler*>(base));
ptr p = { boost::asio::detail::addressof(h->handler_), h, h };

BOOST_ASIO_HANDLER_COMPLETION((*h));

handler_work<Handler, IoExecutor> w(
BOOST_ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
h->work_));

Handler handler(BOOST_ASIO_MOVE_CAST(Handler)(h->handler_));
p.h = boost::asio::detail::addressof(handler);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
BOOST_ASIO_HANDLER_INVOCATION_BEGIN(());
w.complete(handler, handler);
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
