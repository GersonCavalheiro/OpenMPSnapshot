
#ifndef ASIO_DETAIL_RESOLVE_QUERY_OP_HPP
#define ASIO_DETAIL_RESOLVE_QUERY_OP_HPP

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
#include "asio/ip/basic_resolver_query.hpp"
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
class resolve_query_op : public resolve_op
{
public:
ASIO_DEFINE_HANDLER_PTR(resolve_query_op);

typedef asio::ip::basic_resolver_query<Protocol> query_type;
typedef asio::ip::basic_resolver_results<Protocol> results_type;

#if defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif

resolve_query_op(socket_ops::weak_cancel_token_type cancel_token,
const query_type& qry, scheduler_impl& sched,
Handler& handler, const IoExecutor& io_ex)
: resolve_op(&resolve_query_op::do_complete),
cancel_token_(cancel_token),
query_(qry),
scheduler_(sched),
handler_(ASIO_MOVE_CAST(Handler)(handler)),
work_(handler_, io_ex),
addrinfo_(0)
{
}

~resolve_query_op()
{
if (addrinfo_)
socket_ops::freeaddrinfo(addrinfo_);
}

static void do_complete(void* owner, operation* base,
const asio::error_code& ,
std::size_t )
{
resolve_query_op* o(static_cast<resolve_query_op*>(base));
ptr p = { asio::detail::addressof(o->handler_), o, o };

if (owner && owner != &o->scheduler_)
{

socket_ops::background_getaddrinfo(o->cancel_token_,
o->query_.host_name().c_str(), o->query_.service_name().c_str(),
o->query_.hints(), &o->addrinfo_, o->ec_);

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
handler(o->handler_, o->ec_, results_type());
p.h = asio::detail::addressof(handler.handler_);
if (o->addrinfo_)
{
handler.arg2_ = results_type::create(o->addrinfo_,
o->query_.host_name(), o->query_.service_name());
}
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
query_type query_;
scheduler_impl& scheduler_;
Handler handler_;
handler_work<Handler, IoExecutor> work_;
asio::detail::addrinfo_type* addrinfo_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
