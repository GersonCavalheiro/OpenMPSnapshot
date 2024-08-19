
#ifndef ASIO_DETAIL_RESOLVER_SERVICE_HPP
#define ASIO_DETAIL_RESOLVER_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS_RUNTIME)

#include "asio/ip/basic_resolver_query.hpp"
#include "asio/ip/basic_resolver_results.hpp"
#include "asio/detail/concurrency_hint.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/resolve_endpoint_op.hpp"
#include "asio/detail/resolve_query_op.hpp"
#include "asio/detail/resolver_service_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol>
class resolver_service :
public execution_context_service_base<resolver_service<Protocol> >,
public resolver_service_base
{
public:
typedef socket_ops::shared_cancel_token_type implementation_type;

typedef typename Protocol::endpoint endpoint_type;

typedef asio::ip::basic_resolver_query<Protocol> query_type;

typedef asio::ip::basic_resolver_results<Protocol> results_type;

resolver_service(execution_context& context)
: execution_context_service_base<resolver_service<Protocol> >(context),
resolver_service_base(context)
{
}

void shutdown()
{
this->base_shutdown();
}

void notify_fork(execution_context::fork_event fork_ev)
{
this->base_notify_fork(fork_ev);
}

results_type resolve(implementation_type&, const query_type& qry,
asio::error_code& ec)
{
asio::detail::addrinfo_type* address_info = 0;

socket_ops::getaddrinfo(qry.host_name().c_str(),
qry.service_name().c_str(), qry.hints(), &address_info, ec);
auto_addrinfo auto_address_info(address_info);

return ec ? results_type() : results_type::create(
address_info, qry.host_name(), qry.service_name());
}

template <typename Handler, typename IoExecutor>
void async_resolve(implementation_type& impl, const query_type& qry,
Handler& handler, const IoExecutor& io_ex)
{
typedef resolve_query_op<Protocol, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl, qry, scheduler_, handler, io_ex);

ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "resolver", &impl, 0, "async_resolve"));

start_resolve_op(p.p);
p.v = p.p = 0;
}

results_type resolve(implementation_type&,
const endpoint_type& endpoint, asio::error_code& ec)
{
char host_name[NI_MAXHOST];
char service_name[NI_MAXSERV];
socket_ops::sync_getnameinfo(endpoint.data(), endpoint.size(),
host_name, NI_MAXHOST, service_name, NI_MAXSERV,
endpoint.protocol().type(), ec);

return ec ? results_type() : results_type::create(
endpoint, host_name, service_name);
}

template <typename Handler, typename IoExecutor>
void async_resolve(implementation_type& impl, const endpoint_type& endpoint,
Handler& handler, const IoExecutor& io_ex)
{
typedef resolve_endpoint_op<Protocol, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(impl, endpoint, scheduler_, handler, io_ex);

ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "resolver", &impl, 0, "async_resolve"));

start_resolve_op(p.p);
p.v = p.p = 0;
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
