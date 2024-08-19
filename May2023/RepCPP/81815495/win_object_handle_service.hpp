
#ifndef ASIO_DETAIL_WIN_OBJECT_HANDLE_SERVICE_HPP
#define ASIO_DETAIL_WIN_OBJECT_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE)

#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/wait_handler.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"

#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_io_context.hpp"
#else 
# include "asio/detail/scheduler.hpp"
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_object_handle_service :
public execution_context_service_base<win_object_handle_service>
{
public:
typedef HANDLE native_handle_type;

class implementation_type
{
public:
implementation_type()
: handle_(INVALID_HANDLE_VALUE),
wait_handle_(INVALID_HANDLE_VALUE),
owner_(0),
next_(0),
prev_(0)
{
}

private:
friend class win_object_handle_service;

native_handle_type handle_;

HANDLE wait_handle_;

op_queue<wait_op> op_queue_;

win_object_handle_service* owner_;

implementation_type* next_;
implementation_type* prev_;
};

ASIO_DECL win_object_handle_service(execution_context& context);

ASIO_DECL void shutdown();

ASIO_DECL void construct(implementation_type& impl);

ASIO_DECL void move_construct(implementation_type& impl,
implementation_type& other_impl);

ASIO_DECL void move_assign(implementation_type& impl,
win_object_handle_service& other_service,
implementation_type& other_impl);

ASIO_DECL void destroy(implementation_type& impl);

ASIO_DECL asio::error_code assign(implementation_type& impl,
const native_handle_type& handle, asio::error_code& ec);

bool is_open(const implementation_type& impl) const
{
return impl.handle_ != INVALID_HANDLE_VALUE && impl.handle_ != 0;
}

ASIO_DECL asio::error_code close(implementation_type& impl,
asio::error_code& ec);

native_handle_type native_handle(const implementation_type& impl) const
{
return impl.handle_;
}

ASIO_DECL asio::error_code cancel(implementation_type& impl,
asio::error_code& ec);

ASIO_DECL void wait(implementation_type& impl,
asio::error_code& ec);

template <typename Handler, typename IoExecutor>
void async_wait(implementation_type& impl,
Handler& handler, const IoExecutor& io_ex)
{
typedef wait_handler<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(handler, io_ex);

ASIO_HANDLER_CREATION((scheduler_.context(), *p.p, "object_handle",
&impl, reinterpret_cast<uintmax_t>(impl.wait_handle_), "async_wait"));

start_wait_op(impl, p.p);
p.v = p.p = 0;
}

private:
ASIO_DECL void start_wait_op(implementation_type& impl, wait_op* op);

ASIO_DECL void register_wait_callback(
implementation_type& impl, mutex::scoped_lock& lock);

static ASIO_DECL VOID CALLBACK wait_callback(
PVOID param, BOOLEAN timeout);

#if defined(ASIO_HAS_IOCP)
typedef class win_iocp_io_context scheduler_impl;
#else
typedef class scheduler scheduler_impl;
#endif
scheduler_impl& scheduler_;

mutex mutex_;

implementation_type* impl_list_;

bool shutdown_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_object_handle_service.ipp"
#endif 

#endif 

#endif 
