
#ifndef ASIO_BASIC_SOCKET_HPP
#define ASIO_BASIC_SOCKET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/any_io_executor.hpp"
#include "asio/detail/config.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/non_const_lvalue.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/post.hpp"
#include "asio/socket_base.hpp"

#if defined(ASIO_WINDOWS_RUNTIME)
# include "asio/detail/null_socket_service.hpp"
#elif defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_socket_service.hpp"
#else
# include "asio/detail/reactive_socket_service.hpp"
#endif

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {

#if !defined(ASIO_BASIC_SOCKET_FWD_DECL)
#define ASIO_BASIC_SOCKET_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_socket;

#endif 


template <typename Protocol, typename Executor>
class basic_socket
: public socket_base
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_socket<Protocol, Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#elif defined(ASIO_WINDOWS_RUNTIME)
typedef typename detail::null_socket_service<
Protocol>::native_handle_type native_handle_type;
#elif defined(ASIO_HAS_IOCP)
typedef typename detail::win_iocp_socket_service<
Protocol>::native_handle_type native_handle_type;
#else
typedef typename detail::reactive_socket_service<
Protocol>::native_handle_type native_handle_type;
#endif

typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;

#if !defined(ASIO_NO_EXTENSIONS)
typedef basic_socket<Protocol, Executor> lowest_layer_type;
#endif 


explicit basic_socket(const executor_type& ex)
: impl_(0, ex)
{
}


template <typename ExecutionContext>
explicit basic_socket(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
}


basic_socket(const executor_type& ex, const protocol_type& protocol)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
asio::detail::throw_error(ec, "open");
}


template <typename ExecutionContext>
basic_socket(ExecutionContext& context, const protocol_type& protocol,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
asio::detail::throw_error(ec, "open");
}


basic_socket(const executor_type& ex, const endpoint_type& endpoint)
: impl_(0, ex)
{
asio::error_code ec;
const protocol_type protocol = endpoint.protocol();
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
asio::detail::throw_error(ec, "open");
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
asio::detail::throw_error(ec, "bind");
}


template <typename ExecutionContext>
basic_socket(ExecutionContext& context, const endpoint_type& endpoint,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
const protocol_type protocol = endpoint.protocol();
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
asio::detail::throw_error(ec, "open");
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
asio::detail::throw_error(ec, "bind");
}


basic_socket(const executor_type& ex, const protocol_type& protocol,
const native_handle_type& native_socket)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_socket, ec);
asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_socket(ExecutionContext& context, const protocol_type& protocol,
const native_handle_type& native_socket,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_socket, ec);
asio::detail::throw_error(ec, "assign");
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_socket(basic_socket&& other) ASIO_NOEXCEPT
: impl_(std::move(other.impl_))
{
}


basic_socket& operator=(basic_socket&& other)
{
impl_ = std::move(other.impl_);
return *this;
}

template <typename Protocol1, typename Executor1>
friend class basic_socket;


template <typename Protocol1, typename Executor1>
basic_socket(basic_socket<Protocol1, Executor1>&& other,
typename constraint<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value
>::type = 0)
: impl_(std::move(other.impl_))
{
}


template <typename Protocol1, typename Executor1>
typename constraint<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value,
basic_socket&
>::type operator=(basic_socket<Protocol1, Executor1> && other)
{
basic_socket tmp(std::move(other));
impl_ = std::move(tmp.impl_);
return *this;
}
#endif 

executor_type get_executor() ASIO_NOEXCEPT
{
return impl_.get_executor();
}

#if !defined(ASIO_NO_EXTENSIONS)

lowest_layer_type& lowest_layer()
{
return *this;
}


const lowest_layer_type& lowest_layer() const
{
return *this;
}
#endif 


void open(const protocol_type& protocol = protocol_type())
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
asio::detail::throw_error(ec, "open");
}


ASIO_SYNC_OP_VOID open(const protocol_type& protocol,
asio::error_code& ec)
{
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void assign(const protocol_type& protocol,
const native_handle_type& native_socket)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_socket, ec);
asio::detail::throw_error(ec, "assign");
}


ASIO_SYNC_OP_VOID assign(const protocol_type& protocol,
const native_handle_type& native_socket, asio::error_code& ec)
{
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_socket, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}

bool is_open() const
{
return impl_.get_service().is_open(impl_.get_implementation());
}


void close()
{
asio::error_code ec;
impl_.get_service().close(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "close");
}


ASIO_SYNC_OP_VOID close(asio::error_code& ec)
{
impl_.get_service().close(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


#if defined(ASIO_MSVC) && (ASIO_MSVC >= 1400) \
&& (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0603)
__declspec(deprecated("This function always fails with "
"operation_not_supported when used on Windows versions "
"prior to Windows 8.1."))
#endif
native_handle_type release()
{
asio::error_code ec;
native_handle_type s = impl_.get_service().release(
impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "release");
return s;
}


#if defined(ASIO_MSVC) && (ASIO_MSVC >= 1400) \
&& (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0603)
__declspec(deprecated("This function always fails with "
"operation_not_supported when used on Windows versions "
"prior to Windows 8.1."))
#endif
native_handle_type release(asio::error_code& ec)
{
return impl_.get_service().release(impl_.get_implementation(), ec);
}


native_handle_type native_handle()
{
return impl_.get_service().native_handle(impl_.get_implementation());
}


#if defined(ASIO_MSVC) && (ASIO_MSVC >= 1400) \
&& (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600) \
&& !defined(ASIO_ENABLE_CANCELIO)
__declspec(deprecated("By default, this function always fails with "
"operation_not_supported when used on Windows XP, Windows Server 2003, "
"or earlier. Consult documentation for details."))
#endif
void cancel()
{
asio::error_code ec;
impl_.get_service().cancel(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "cancel");
}


#if defined(ASIO_MSVC) && (ASIO_MSVC >= 1400) \
&& (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0600) \
&& !defined(ASIO_ENABLE_CANCELIO)
__declspec(deprecated("By default, this function always fails with "
"operation_not_supported when used on Windows XP, Windows Server 2003, "
"or earlier. Consult documentation for details."))
#endif
ASIO_SYNC_OP_VOID cancel(asio::error_code& ec)
{
impl_.get_service().cancel(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool at_mark() const
{
asio::error_code ec;
bool b = impl_.get_service().at_mark(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "at_mark");
return b;
}


bool at_mark(asio::error_code& ec) const
{
return impl_.get_service().at_mark(impl_.get_implementation(), ec);
}


std::size_t available() const
{
asio::error_code ec;
std::size_t s = impl_.get_service().available(
impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "available");
return s;
}


std::size_t available(asio::error_code& ec) const
{
return impl_.get_service().available(impl_.get_implementation(), ec);
}


void bind(const endpoint_type& endpoint)
{
asio::error_code ec;
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
asio::detail::throw_error(ec, "bind");
}


ASIO_SYNC_OP_VOID bind(const endpoint_type& endpoint,
asio::error_code& ec)
{
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void connect(const endpoint_type& peer_endpoint)
{
asio::error_code ec;
if (!is_open())
{
impl_.get_service().open(impl_.get_implementation(),
peer_endpoint.protocol(), ec);
asio::detail::throw_error(ec, "connect");
}
impl_.get_service().connect(impl_.get_implementation(), peer_endpoint, ec);
asio::detail::throw_error(ec, "connect");
}


ASIO_SYNC_OP_VOID connect(const endpoint_type& peer_endpoint,
asio::error_code& ec)
{
if (!is_open())
{
impl_.get_service().open(impl_.get_implementation(),
peer_endpoint.protocol(), ec);
if (ec)
{
ASIO_SYNC_OP_VOID_RETURN(ec);
}
}

impl_.get_service().connect(impl_.get_implementation(), peer_endpoint, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code))
ConnectHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ConnectHandler,
void (asio::error_code))
async_connect(const endpoint_type& peer_endpoint,
ASIO_MOVE_ARG(ConnectHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
asio::error_code open_ec;
if (!is_open())
{
const protocol_type protocol = peer_endpoint.protocol();
impl_.get_service().open(impl_.get_implementation(), protocol, open_ec);
}

return async_initiate<ConnectHandler, void (asio::error_code)>(
initiate_async_connect(this), handler, peer_endpoint, open_ec);
}


template <typename SettableSocketOption>
void set_option(const SettableSocketOption& option)
{
asio::error_code ec;
impl_.get_service().set_option(impl_.get_implementation(), option, ec);
asio::detail::throw_error(ec, "set_option");
}


template <typename SettableSocketOption>
ASIO_SYNC_OP_VOID set_option(const SettableSocketOption& option,
asio::error_code& ec)
{
impl_.get_service().set_option(impl_.get_implementation(), option, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename GettableSocketOption>
void get_option(GettableSocketOption& option) const
{
asio::error_code ec;
impl_.get_service().get_option(impl_.get_implementation(), option, ec);
asio::detail::throw_error(ec, "get_option");
}


template <typename GettableSocketOption>
ASIO_SYNC_OP_VOID get_option(GettableSocketOption& option,
asio::error_code& ec) const
{
impl_.get_service().get_option(impl_.get_implementation(), option, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename IoControlCommand>
void io_control(IoControlCommand& command)
{
asio::error_code ec;
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
asio::detail::throw_error(ec, "io_control");
}


template <typename IoControlCommand>
ASIO_SYNC_OP_VOID io_control(IoControlCommand& command,
asio::error_code& ec)
{
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool non_blocking() const
{
return impl_.get_service().non_blocking(impl_.get_implementation());
}


void non_blocking(bool mode)
{
asio::error_code ec;
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
asio::detail::throw_error(ec, "non_blocking");
}


ASIO_SYNC_OP_VOID non_blocking(
bool mode, asio::error_code& ec)
{
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool native_non_blocking() const
{
return impl_.get_service().native_non_blocking(impl_.get_implementation());
}


void native_non_blocking(bool mode)
{
asio::error_code ec;
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
asio::detail::throw_error(ec, "native_non_blocking");
}


ASIO_SYNC_OP_VOID native_non_blocking(
bool mode, asio::error_code& ec)
{
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


endpoint_type local_endpoint() const
{
asio::error_code ec;
endpoint_type ep = impl_.get_service().local_endpoint(
impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "local_endpoint");
return ep;
}


endpoint_type local_endpoint(asio::error_code& ec) const
{
return impl_.get_service().local_endpoint(impl_.get_implementation(), ec);
}


endpoint_type remote_endpoint() const
{
asio::error_code ec;
endpoint_type ep = impl_.get_service().remote_endpoint(
impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "remote_endpoint");
return ep;
}


endpoint_type remote_endpoint(asio::error_code& ec) const
{
return impl_.get_service().remote_endpoint(impl_.get_implementation(), ec);
}


void shutdown(shutdown_type what)
{
asio::error_code ec;
impl_.get_service().shutdown(impl_.get_implementation(), what, ec);
asio::detail::throw_error(ec, "shutdown");
}


ASIO_SYNC_OP_VOID shutdown(shutdown_type what,
asio::error_code& ec)
{
impl_.get_service().shutdown(impl_.get_implementation(), what, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void wait(wait_type w)
{
asio::error_code ec;
impl_.get_service().wait(impl_.get_implementation(), w, ec);
asio::detail::throw_error(ec, "wait");
}


ASIO_SYNC_OP_VOID wait(wait_type w, asio::error_code& ec)
{
impl_.get_service().wait(impl_.get_implementation(), w, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code))
WaitHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler,
void (asio::error_code))
async_wait(wait_type w,
ASIO_MOVE_ARG(WaitHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WaitHandler, void (asio::error_code)>(
initiate_async_wait(this), handler, w);
}

protected:

~basic_socket()
{
}

#if defined(ASIO_WINDOWS_RUNTIME)
detail::io_object_impl<
detail::null_socket_service<Protocol>, Executor> impl_;
#elif defined(ASIO_HAS_IOCP)
detail::io_object_impl<
detail::win_iocp_socket_service<Protocol>, Executor> impl_;
#else
detail::io_object_impl<
detail::reactive_socket_service<Protocol>, Executor> impl_;
#endif

private:
basic_socket(const basic_socket&) ASIO_DELETED;
basic_socket& operator=(const basic_socket&) ASIO_DELETED;

class initiate_async_connect
{
public:
typedef Executor executor_type;

explicit initiate_async_connect(basic_socket* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ConnectHandler>
void operator()(ASIO_MOVE_ARG(ConnectHandler) handler,
const endpoint_type& peer_endpoint,
const asio::error_code& open_ec) const
{
ASIO_CONNECT_HANDLER_CHECK(ConnectHandler, handler) type_check;

if (open_ec)
{
asio::post(self_->impl_.get_executor(),
asio::detail::bind_handler(
ASIO_MOVE_CAST(ConnectHandler)(handler), open_ec));
}
else
{
detail::non_const_lvalue<ConnectHandler> handler2(handler);
self_->impl_.get_service().async_connect(
self_->impl_.get_implementation(), peer_endpoint,
handler2.value, self_->impl_.get_executor());
}
}

private:
basic_socket* self_;
};

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_socket* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WaitHandler>
void operator()(ASIO_MOVE_ARG(WaitHandler) handler, wait_type w) const
{
ASIO_WAIT_HANDLER_CHECK(WaitHandler, handler) type_check;

detail::non_const_lvalue<WaitHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(), w,
handler2.value, self_->impl_.get_executor());
}

private:
basic_socket* self_;
};
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
