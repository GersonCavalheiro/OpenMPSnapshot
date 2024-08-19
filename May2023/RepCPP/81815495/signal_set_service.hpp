
#ifndef ASIO_DETAIL_SIGNAL_SET_SERVICE_HPP
#define ASIO_DETAIL_SIGNAL_SET_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include <cstddef>
#include <signal.h>
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/signal_handler.hpp"
#include "asio/detail/signal_op.hpp"
#include "asio/detail/socket_types.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#else 
# include "asio/detail/scheduler.hpp"
#endif 

#if !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)
# include "asio/detail/reactor.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

#if defined(NSIG) && (NSIG > 0)
enum { max_signal_number = NSIG };
#else
enum { max_signal_number = 128 };
#endif

extern ASIO_DECL struct signal_state* get_signal_state();

extern "C" ASIO_DECL void asio_signal_handler(int signal_number);

class signal_set_service :
public execution_context_service_base<signal_set_service>
{
public:
class registration
{
public:
registration()
: signal_number_(0),
queue_(0),
undelivered_(0),
next_in_table_(0),
prev_in_table_(0),
next_in_set_(0)
{
}

private:
friend class signal_set_service;

int signal_number_;

op_queue<signal_op>* queue_;

std::size_t undelivered_;

registration* next_in_table_;
registration* prev_in_table_;

registration* next_in_set_;
};

class implementation_type
{
public:
implementation_type()
: signals_(0)
{
}

private:
friend class signal_set_service;

op_queue<signal_op> queue_;

registration* signals_;
};

ASIO_DECL signal_set_service(execution_context& context);

ASIO_DECL ~signal_set_service();

ASIO_DECL void shutdown();

ASIO_DECL void notify_fork(
asio::execution_context::fork_event fork_ev);

ASIO_DECL void construct(implementation_type& impl);

ASIO_DECL void destroy(implementation_type& impl);

ASIO_DECL asio::error_code add(implementation_type& impl,
int signal_number, asio::error_code& ec);

ASIO_DECL asio::error_code remove(implementation_type& impl,
int signal_number, asio::error_code& ec);

ASIO_DECL asio::error_code clear(implementation_type& impl,
asio::error_code& ec);

ASIO_DECL asio::error_code cancel(implementation_type& impl,
asio::error_code& ec);

template <typename Handler, typename IoExecutor>
void async_wait(implementation_type& impl,
Handler& handler, const IoExecutor& io_ex)
{
typedef signal_handler<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(handler, io_ex);

ASIO_HANDLER_CREATION((scheduler_.context(),
*p.p, "signal_set", &impl, 0, "async_wait"));

start_wait_op(impl, p.p);
p.v = p.p = 0;
}

ASIO_DECL static void deliver_signal(int signal_number);

private:
ASIO_DECL static void add_service(signal_set_service* service);

ASIO_DECL static void remove_service(signal_set_service* service);

ASIO_DECL static void open_descriptors();

ASIO_DECL static void close_descriptors();

ASIO_DECL void start_wait_op(implementation_type& impl, signal_op* op);

#if defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif
scheduler_impl& scheduler_;

#if !defined(ASIO_WINDOWS) \
&& !defined(ASIO_WINDOWS_RUNTIME) \
&& !defined(__CYGWIN__)
class pipe_read_op;

reactor& reactor_;

reactor::per_descriptor_data reactor_data_;
#endif 

registration* registrations_[max_signal_number];

signal_set_service* next_;
signal_set_service* prev_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/signal_set_service.ipp"
#endif 

#endif 
