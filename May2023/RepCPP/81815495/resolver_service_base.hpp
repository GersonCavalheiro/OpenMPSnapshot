
#ifndef ASIO_DETAIL_RESOLVER_SERVICE_BASE_HPP
#define ASIO_DETAIL_RESOLVER_SERVICE_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/resolve_op.hpp"
#include "asio/detail/socket_ops.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/detail/scoped_ptr.hpp"
#include "asio/detail/thread.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#else 
# include "asio/detail/scheduler.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class resolver_service_base
{
public:
typedef socket_ops::shared_cancel_token_type implementation_type;

ASIO_DECL resolver_service_base(execution_context& context);

ASIO_DECL ~resolver_service_base();

ASIO_DECL void base_shutdown();

ASIO_DECL void base_notify_fork(
execution_context::fork_event fork_ev);

ASIO_DECL void construct(implementation_type& impl);

ASIO_DECL void destroy(implementation_type&);

ASIO_DECL void move_construct(implementation_type& impl,
implementation_type& other_impl);

ASIO_DECL void move_assign(implementation_type& impl,
resolver_service_base& other_service,
implementation_type& other_impl);

void converting_move_construct(implementation_type& impl,
resolver_service_base&, implementation_type& other_impl)
{
move_construct(impl, other_impl);
}

void converting_move_assign(implementation_type& impl,
resolver_service_base& other_service,
implementation_type& other_impl)
{
move_assign(impl, other_service, other_impl);
}

ASIO_DECL void cancel(implementation_type& impl);

protected:
ASIO_DECL void start_resolve_op(resolve_op* op);

#if !defined(ASIO_WINDOWS_RUNTIME)
class auto_addrinfo
: private asio::detail::noncopyable
{
public:
explicit auto_addrinfo(asio::detail::addrinfo_type* ai)
: ai_(ai)
{
}

~auto_addrinfo()
{
if (ai_)
socket_ops::freeaddrinfo(ai_);
}

operator asio::detail::addrinfo_type*()
{
return ai_;
}

private:
asio::detail::addrinfo_type* ai_;
};
#endif 

class work_scheduler_runner;

ASIO_DECL void start_work_thread();

#if defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif
scheduler_impl& scheduler_;

private:
asio::detail::mutex mutex_;

asio::detail::scoped_ptr<scheduler_impl> work_scheduler_;

asio::detail::scoped_ptr<asio::detail::thread> work_thread_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/resolver_service_base.ipp"
#endif 

#endif 
