
#ifndef ASIO_DETAIL_RESOLVER_ENDPOINT_OP_HPP
#define ASIO_DETAIL_RESOLVER_ENDPOINT_OP_HPP

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
#include "asio/detail/resolve_op.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/error.hpp"
#include "asio/ip/basic_resolver_results.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#else 
# include "asio/detail/scheduler.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol, typename Handler, typename IoExecutor>
class resolve_endpoint_op : public resolve_op
{
public:
ASIO_DEFINE_HANDLER_PTR(resolve_endpoint_op);

typedef typename Protocol::endpoint endpoint_type;
typedef asio::ip::basic_resolver_results<Protocol> results_type;

#if defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif

resolve_endpoint_op(socket_ops::weak_cancel_token_type cancel_token,
const endpoint_type& endpoint, scheduler_impl& sched,
Handler& handler, const IoExecutor& io_ex)
: resolve_op(&resolve_endpoint_op::do_complete),
cancel_token_(cancel_token),
endpoint_(endpoint),
scheduler_(sched),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& ,
std::size_t )
{
resolve_endpoint_op* o(static_cast<resolve_endpoint_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

if (owner && owner != &o->scheduler_)
{

char host_name[NI_MAXHOST] = "";
char service_name[NI_MAXSERV] = "";
socket_ops::background_getnameinfo(o->cancel_token_, o->endpoint_.data(),
o->endpoint_.size(), host_name, NI_MAXHOST, service_name, NI_MAXSERV,
o->endpoint_.protocol().type(), o->ec_);
o->results_ = results_type::create(o->endpoint_, host_name, service_name);

o->scheduler_.post_deferred_completion(o);
p.v = p.p = 0;
}
else
{

ASIO_HANDLER_COMPLETION((*o));

handler_work<Handler, IoExecutor> w(
ASIO_MOVE_CAST2(handler_work<Handler, IoExecutor>)(
o->work_));

detail::binder2<Handler, asio::error_code, results_type>
handler(o->handler_, o->ec_, o->results_);
p.h = asio::detail::addressof(handler.handler_);
p.reset();

if (owner)
{
fenced_block b(fenced_block::half);
ASIO_HANDLER_INVOCATION_BEGIN((handler.arg1_, "..."));
w.complete(handler, handler.handler_);
ASIO_HANDLER_INVOCATION_END;
}
}
}

private:
socket_ops::weak_cancel_token_type cancel_token_;
endpoint_type endpoint_;
scheduler_impl& scheduler_;
Handler handler_;
handler_work<Handler, IoExecutor> work_;
results_type results_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
