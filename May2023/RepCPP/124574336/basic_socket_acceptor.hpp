
#ifndef BOOST_ASIO_BASIC_SOCKET_ACCEPTOR_HPP
#define BOOST_ASIO_BASIC_SOCKET_ACCEPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/basic_socket.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/io_object_impl.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/socket_base.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
# include <boost/asio/detail/null_socket_service.hpp>
#elif defined(BOOST_ASIO_HAS_IOCP)
# include <boost/asio/detail/win_iocp_socket_service.hpp>
#else
# include <boost/asio/detail/reactive_socket_service.hpp>
#endif

#if defined(BOOST_ASIO_HAS_MOVE)
# include <utility>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if !defined(BOOST_ASIO_BASIC_SOCKET_ACCEPTOR_FWD_DECL)
#define BOOST_ASIO_BASIC_SOCKET_ACCEPTOR_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_socket_acceptor;

#endif 


template <typename Protocol, typename Executor>
class basic_socket_acceptor
: public socket_base
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_socket_acceptor<Protocol, Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#elif defined(BOOST_ASIO_WINDOWS_RUNTIME)
typedef typename detail::null_socket_service<
Protocol>::native_handle_type native_handle_type;
#elif defined(BOOST_ASIO_HAS_IOCP)
typedef typename detail::win_iocp_socket_service<
Protocol>::native_handle_type native_handle_type;
#else
typedef typename detail::reactive_socket_service<
Protocol>::native_handle_type native_handle_type;
#endif

typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;


explicit basic_socket_acceptor(const executor_type& ex)
: impl_(ex)
{
}


template <typename ExecutionContext>
explicit basic_socket_acceptor(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
}


basic_socket_acceptor(const executor_type& ex, const protocol_type& protocol)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
boost::asio::detail::throw_error(ec, "open");
}


template <typename ExecutionContext>
basic_socket_acceptor(ExecutionContext& context,
const protocol_type& protocol,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
boost::asio::detail::throw_error(ec, "open");
}


basic_socket_acceptor(const executor_type& ex,
const endpoint_type& endpoint, bool reuse_addr = true)
: impl_(ex)
{
boost::system::error_code ec;
const protocol_type protocol = endpoint.protocol();
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
boost::asio::detail::throw_error(ec, "open");
if (reuse_addr)
{
impl_.get_service().set_option(impl_.get_implementation(),
socket_base::reuse_address(true), ec);
boost::asio::detail::throw_error(ec, "set_option");
}
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
boost::asio::detail::throw_error(ec, "bind");
impl_.get_service().listen(impl_.get_implementation(),
socket_base::max_listen_connections, ec);
boost::asio::detail::throw_error(ec, "listen");
}


template <typename ExecutionContext>
basic_socket_acceptor(ExecutionContext& context,
const endpoint_type& endpoint, bool reuse_addr = true,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
const protocol_type protocol = endpoint.protocol();
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
boost::asio::detail::throw_error(ec, "open");
if (reuse_addr)
{
impl_.get_service().set_option(impl_.get_implementation(),
socket_base::reuse_address(true), ec);
boost::asio::detail::throw_error(ec, "set_option");
}
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
boost::asio::detail::throw_error(ec, "bind");
impl_.get_service().listen(impl_.get_implementation(),
socket_base::max_listen_connections, ec);
boost::asio::detail::throw_error(ec, "listen");
}


basic_socket_acceptor(const executor_type& ex,
const protocol_type& protocol, const native_handle_type& native_acceptor)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_acceptor, ec);
boost::asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_socket_acceptor(ExecutionContext& context,
const protocol_type& protocol, const native_handle_type& native_acceptor,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_acceptor, ec);
boost::asio::detail::throw_error(ec, "assign");
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_socket_acceptor(basic_socket_acceptor&& other) BOOST_ASIO_NOEXCEPT
: impl_(std::move(other.impl_))
{
}


basic_socket_acceptor& operator=(basic_socket_acceptor&& other)
{
impl_ = std::move(other.impl_);
return *this;
}

template <typename Protocol1, typename Executor1>
friend class basic_socket_acceptor;


template <typename Protocol1, typename Executor1>
basic_socket_acceptor(basic_socket_acceptor<Protocol1, Executor1>&& other,
typename enable_if<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value
>::type* = 0)
: impl_(std::move(other.impl_))
{
}


template <typename Protocol1, typename Executor1>
typename enable_if<
is_convertible<Protocol1, Protocol>::value
&& is_convertible<Executor1, Executor>::value,
basic_socket_acceptor&
>::type operator=(basic_socket_acceptor<Protocol1, Executor1>&& other)
{
basic_socket_acceptor tmp(std::move(other));
impl_ = std::move(tmp.impl_);
return *this;
}
#endif 


~basic_socket_acceptor()
{
}

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return impl_.get_executor();
}


void open(const protocol_type& protocol = protocol_type())
{
boost::system::error_code ec;
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
boost::asio::detail::throw_error(ec, "open");
}


BOOST_ASIO_SYNC_OP_VOID open(const protocol_type& protocol,
boost::system::error_code& ec)
{
impl_.get_service().open(impl_.get_implementation(), protocol, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void assign(const protocol_type& protocol,
const native_handle_type& native_acceptor)
{
boost::system::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_acceptor, ec);
boost::asio::detail::throw_error(ec, "assign");
}


BOOST_ASIO_SYNC_OP_VOID assign(const protocol_type& protocol,
const native_handle_type& native_acceptor, boost::system::error_code& ec)
{
impl_.get_service().assign(impl_.get_implementation(),
protocol, native_acceptor, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}

bool is_open() const
{
return impl_.get_service().is_open(impl_.get_implementation());
}


void bind(const endpoint_type& endpoint)
{
boost::system::error_code ec;
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
boost::asio::detail::throw_error(ec, "bind");
}


BOOST_ASIO_SYNC_OP_VOID bind(const endpoint_type& endpoint,
boost::system::error_code& ec)
{
impl_.get_service().bind(impl_.get_implementation(), endpoint, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void listen(int backlog = socket_base::max_listen_connections)
{
boost::system::error_code ec;
impl_.get_service().listen(impl_.get_implementation(), backlog, ec);
boost::asio::detail::throw_error(ec, "listen");
}


BOOST_ASIO_SYNC_OP_VOID listen(int backlog, boost::system::error_code& ec)
{
impl_.get_service().listen(impl_.get_implementation(), backlog, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void close()
{
boost::system::error_code ec;
impl_.get_service().close(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "close");
}


BOOST_ASIO_SYNC_OP_VOID close(boost::system::error_code& ec)
{
impl_.get_service().close(impl_.get_implementation(), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


#if defined(BOOST_ASIO_MSVC) && (BOOST_ASIO_MSVC >= 1400) \
&& (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0603)
__declspec(deprecated("This function always fails with "
"operation_not_supported when used on Windows versions "
"prior to Windows 8.1."))
#endif
native_handle_type release()
{
boost::system::error_code ec;
native_handle_type s = impl_.get_service().release(
impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "release");
return s;
}


#if defined(BOOST_ASIO_MSVC) && (BOOST_ASIO_MSVC >= 1400) \
&& (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0603)
__declspec(deprecated("This function always fails with "
"operation_not_supported when used on Windows versions "
"prior to Windows 8.1."))
#endif
native_handle_type release(boost::system::error_code& ec)
{
return impl_.get_service().release(impl_.get_implementation(), ec);
}


native_handle_type native_handle()
{
return impl_.get_service().native_handle(impl_.get_implementation());
}


void cancel()
{
boost::system::error_code ec;
impl_.get_service().cancel(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "cancel");
}


BOOST_ASIO_SYNC_OP_VOID cancel(boost::system::error_code& ec)
{
impl_.get_service().cancel(impl_.get_implementation(), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename SettableSocketOption>
void set_option(const SettableSocketOption& option)
{
boost::system::error_code ec;
impl_.get_service().set_option(impl_.get_implementation(), option, ec);
boost::asio::detail::throw_error(ec, "set_option");
}


template <typename SettableSocketOption>
BOOST_ASIO_SYNC_OP_VOID set_option(const SettableSocketOption& option,
boost::system::error_code& ec)
{
impl_.get_service().set_option(impl_.get_implementation(), option, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename GettableSocketOption>
void get_option(GettableSocketOption& option) const
{
boost::system::error_code ec;
impl_.get_service().get_option(impl_.get_implementation(), option, ec);
boost::asio::detail::throw_error(ec, "get_option");
}


template <typename GettableSocketOption>
BOOST_ASIO_SYNC_OP_VOID get_option(GettableSocketOption& option,
boost::system::error_code& ec) const
{
impl_.get_service().get_option(impl_.get_implementation(), option, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename IoControlCommand>
void io_control(IoControlCommand& command)
{
boost::system::error_code ec;
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
boost::asio::detail::throw_error(ec, "io_control");
}


template <typename IoControlCommand>
BOOST_ASIO_SYNC_OP_VOID io_control(IoControlCommand& command,
boost::system::error_code& ec)
{
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool non_blocking() const
{
return impl_.get_service().non_blocking(impl_.get_implementation());
}


void non_blocking(bool mode)
{
boost::system::error_code ec;
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
boost::asio::detail::throw_error(ec, "non_blocking");
}


BOOST_ASIO_SYNC_OP_VOID non_blocking(
bool mode, boost::system::error_code& ec)
{
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool native_non_blocking() const
{
return impl_.get_service().native_non_blocking(impl_.get_implementation());
}


void native_non_blocking(bool mode)
{
boost::system::error_code ec;
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
boost::asio::detail::throw_error(ec, "native_non_blocking");
}


BOOST_ASIO_SYNC_OP_VOID native_non_blocking(
bool mode, boost::system::error_code& ec)
{
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


endpoint_type local_endpoint() const
{
boost::system::error_code ec;
endpoint_type ep = impl_.get_service().local_endpoint(
impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "local_endpoint");
return ep;
}


endpoint_type local_endpoint(boost::system::error_code& ec) const
{
return impl_.get_service().local_endpoint(impl_.get_implementation(), ec);
}


void wait(wait_type w)
{
boost::system::error_code ec;
impl_.get_service().wait(impl_.get_implementation(), w, ec);
boost::asio::detail::throw_error(ec, "wait");
}


BOOST_ASIO_SYNC_OP_VOID wait(wait_type w, boost::system::error_code& ec)
{
impl_.get_service().wait(impl_.get_implementation(), w, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code))
WaitHandler BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler,
void (boost::system::error_code))
async_wait(wait_type w,
BOOST_ASIO_MOVE_ARG(WaitHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WaitHandler, void (boost::system::error_code)>(
initiate_async_wait(this), handler, w);
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)

template <typename Protocol1, typename Executor1>
void accept(basic_socket<Protocol1, Executor1>& peer,
typename enable_if<
is_convertible<Protocol, Protocol1>::value
>::type* = 0)
{
boost::system::error_code ec;
impl_.get_service().accept(impl_.get_implementation(),
peer, static_cast<endpoint_type*>(0), ec);
boost::asio::detail::throw_error(ec, "accept");
}


template <typename Protocol1, typename Executor1>
BOOST_ASIO_SYNC_OP_VOID accept(
basic_socket<Protocol1, Executor1>& peer, boost::system::error_code& ec,
typename enable_if<
is_convertible<Protocol, Protocol1>::value
>::type* = 0)
{
impl_.get_service().accept(impl_.get_implementation(),
peer, static_cast<endpoint_type*>(0), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename Protocol1, typename Executor1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code))
AcceptHandler BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(AcceptHandler,
void (boost::system::error_code))
async_accept(basic_socket<Protocol1, Executor1>& peer,
BOOST_ASIO_MOVE_ARG(AcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type),
typename enable_if<
is_convertible<Protocol, Protocol1>::value
>::type* = 0)
{
return async_initiate<AcceptHandler, void (boost::system::error_code)>(
initiate_async_accept(this), handler,
&peer, static_cast<endpoint_type*>(0));
}


template <typename Executor1>
void accept(basic_socket<protocol_type, Executor1>& peer,
endpoint_type& peer_endpoint)
{
boost::system::error_code ec;
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
boost::asio::detail::throw_error(ec, "accept");
}


template <typename Executor1>
BOOST_ASIO_SYNC_OP_VOID accept(basic_socket<protocol_type, Executor1>& peer,
endpoint_type& peer_endpoint, boost::system::error_code& ec)
{
impl_.get_service().accept(
impl_.get_implementation(), peer, &peer_endpoint, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename Executor1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code))
AcceptHandler BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(AcceptHandler,
void (boost::system::error_code))
async_accept(basic_socket<protocol_type, Executor1>& peer,
endpoint_type& peer_endpoint,
BOOST_ASIO_MOVE_ARG(AcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<AcceptHandler, void (boost::system::error_code)>(
initiate_async_accept(this), handler, &peer, &peer_endpoint);
}
#endif 

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

typename Protocol::socket::template rebind_executor<executor_type>::other
accept()
{
boost::system::error_code ec;
typename Protocol::socket::template rebind_executor<
executor_type>::other peer(impl_.get_executor());
impl_.get_service().accept(impl_.get_implementation(), peer, 0, ec);
boost::asio::detail::throw_error(ec, "accept");
return peer;
}


typename Protocol::socket::template rebind_executor<executor_type>::other
accept(boost::system::error_code& ec)
{
typename Protocol::socket::template rebind_executor<
executor_type>::other peer(impl_.get_executor());
impl_.get_service().accept(impl_.get_implementation(), peer, 0, ec);
return peer;
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
executor_type>::other)) MoveAcceptHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(MoveAcceptHandler,
void (boost::system::error_code,
typename Protocol::socket::template
rebind_executor<executor_type>::other))
async_accept(
BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<MoveAcceptHandler,
void (boost::system::error_code, typename Protocol::socket::template
rebind_executor<executor_type>::other)>(
initiate_async_move_accept(this), handler,
impl_.get_executor(), static_cast<endpoint_type*>(0),
static_cast<typename Protocol::socket::template
rebind_executor<executor_type>::other*>(0));
}


template <typename Executor1>
typename Protocol::socket::template rebind_executor<Executor1>::other
accept(const Executor1& ex,
typename enable_if<
is_executor<Executor1>::value
|| execution::is_executor<Executor1>::value
>::type* = 0)
{
boost::system::error_code ec;
typename Protocol::socket::template
rebind_executor<Executor1>::other peer(ex);
impl_.get_service().accept(impl_.get_implementation(), peer, 0, ec);
boost::asio::detail::throw_error(ec, "accept");
return peer;
}


template <typename ExecutionContext>
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other
accept(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
boost::system::error_code ec;
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other peer(context);
impl_.get_service().accept(impl_.get_implementation(), peer, 0, ec);
boost::asio::detail::throw_error(ec, "accept");
return peer;
}


template <typename Executor1>
typename Protocol::socket::template rebind_executor<Executor1>::other
accept(const Executor1& ex, boost::system::error_code& ec,
typename enable_if<
is_executor<Executor1>::value
|| execution::is_executor<Executor1>::value
>::type* = 0)
{
typename Protocol::socket::template
rebind_executor<Executor1>::other peer(ex);
impl_.get_service().accept(impl_.get_implementation(), peer, 0, ec);
return peer;
}


template <typename ExecutionContext>
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other
accept(ExecutionContext& context, boost::system::error_code& ec,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other peer(context);
impl_.get_service().accept(impl_.get_implementation(), peer, 0, ec);
return peer;
}


template <typename Executor1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
Executor1>::other)) MoveAcceptHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(MoveAcceptHandler,
void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
Executor1>::other))
async_accept(const Executor1& ex,
BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type),
typename enable_if<
is_executor<Executor1>::value
|| execution::is_executor<Executor1>::value
>::type* = 0)
{
typedef typename Protocol::socket::template rebind_executor<
Executor1>::other other_socket_type;

return async_initiate<MoveAcceptHandler,
void (boost::system::error_code, other_socket_type)>(
initiate_async_move_accept(this), handler,
ex, static_cast<endpoint_type*>(0),
static_cast<other_socket_type*>(0));
}


template <typename ExecutionContext,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other)) MoveAcceptHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(MoveAcceptHandler,
void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other))
async_accept(ExecutionContext& context,
BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type),
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
typedef typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other other_socket_type;

return async_initiate<MoveAcceptHandler,
void (boost::system::error_code, other_socket_type)>(
initiate_async_move_accept(this), handler,
context.get_executor(), static_cast<endpoint_type*>(0),
static_cast<other_socket_type*>(0));
}


typename Protocol::socket::template rebind_executor<executor_type>::other
accept(endpoint_type& peer_endpoint)
{
boost::system::error_code ec;
typename Protocol::socket::template rebind_executor<
executor_type>::other peer(impl_.get_executor());
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
boost::asio::detail::throw_error(ec, "accept");
return peer;
}


typename Protocol::socket::template rebind_executor<executor_type>::other
accept(endpoint_type& peer_endpoint, boost::system::error_code& ec)
{
typename Protocol::socket::template rebind_executor<
executor_type>::other peer(impl_.get_executor());
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
return peer;
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
executor_type>::other)) MoveAcceptHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(MoveAcceptHandler,
void (boost::system::error_code,
typename Protocol::socket::template
rebind_executor<executor_type>::other))
async_accept(endpoint_type& peer_endpoint,
BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<MoveAcceptHandler,
void (boost::system::error_code, typename Protocol::socket::template
rebind_executor<executor_type>::other)>(
initiate_async_move_accept(this), handler,
impl_.get_executor(), &peer_endpoint,
static_cast<typename Protocol::socket::template
rebind_executor<executor_type>::other*>(0));
}


template <typename Executor1>
typename Protocol::socket::template rebind_executor<Executor1>::other
accept(const Executor1& ex, endpoint_type& peer_endpoint,
typename enable_if<
is_executor<Executor1>::value
|| execution::is_executor<Executor1>::value
>::type* = 0)
{
boost::system::error_code ec;
typename Protocol::socket::template
rebind_executor<Executor1>::other peer(ex);
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
boost::asio::detail::throw_error(ec, "accept");
return peer;
}


template <typename ExecutionContext>
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other
accept(ExecutionContext& context, endpoint_type& peer_endpoint,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
boost::system::error_code ec;
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other peer(context);
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
boost::asio::detail::throw_error(ec, "accept");
return peer;
}


template <typename Executor1>
typename Protocol::socket::template rebind_executor<Executor1>::other
accept(const executor_type& ex,
endpoint_type& peer_endpoint, boost::system::error_code& ec,
typename enable_if<
is_executor<Executor1>::value
|| execution::is_executor<Executor1>::value
>::type* = 0)
{
typename Protocol::socket::template
rebind_executor<Executor1>::other peer(ex);
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
return peer;
}


template <typename ExecutionContext>
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other
accept(ExecutionContext& context,
endpoint_type& peer_endpoint, boost::system::error_code& ec,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other peer(context);
impl_.get_service().accept(impl_.get_implementation(),
peer, &peer_endpoint, ec);
return peer;
}


template <typename Executor1,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
Executor1>::other)) MoveAcceptHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(MoveAcceptHandler,
void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
Executor1>::other))
async_accept(const Executor1& ex, endpoint_type& peer_endpoint,
BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type),
typename enable_if<
is_executor<Executor1>::value
|| execution::is_executor<Executor1>::value
>::type* = 0)
{
typedef typename Protocol::socket::template rebind_executor<
Executor1>::other other_socket_type;

return async_initiate<MoveAcceptHandler,
void (boost::system::error_code, other_socket_type)>(
initiate_async_move_accept(this), handler,
ex, &peer_endpoint,
static_cast<other_socket_type*>(0));
}


template <typename ExecutionContext,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other)) MoveAcceptHandler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(MoveAcceptHandler,
void (boost::system::error_code,
typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other))
async_accept(ExecutionContext& context,
endpoint_type& peer_endpoint,
BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type),
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
{
typedef typename Protocol::socket::template rebind_executor<
typename ExecutionContext::executor_type>::other other_socket_type;

return async_initiate<MoveAcceptHandler,
void (boost::system::error_code, other_socket_type)>(
initiate_async_move_accept(this), handler,
context.get_executor(), &peer_endpoint,
static_cast<other_socket_type*>(0));
}
#endif 

private:
basic_socket_acceptor(const basic_socket_acceptor&) BOOST_ASIO_DELETED;
basic_socket_acceptor& operator=(
const basic_socket_acceptor&) BOOST_ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_socket_acceptor* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WaitHandler>
void operator()(BOOST_ASIO_MOVE_ARG(WaitHandler) handler, wait_type w) const
{
BOOST_ASIO_WAIT_HANDLER_CHECK(WaitHandler, handler) type_check;

detail::non_const_lvalue<WaitHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(), w,
handler2.value, self_->impl_.get_executor());
}

private:
basic_socket_acceptor* self_;
};

class initiate_async_accept
{
public:
typedef Executor executor_type;

explicit initiate_async_accept(basic_socket_acceptor* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename AcceptHandler, typename Protocol1, typename Executor1>
void operator()(BOOST_ASIO_MOVE_ARG(AcceptHandler) handler,
basic_socket<Protocol1, Executor1>* peer,
endpoint_type* peer_endpoint) const
{
BOOST_ASIO_ACCEPT_HANDLER_CHECK(AcceptHandler, handler) type_check;

detail::non_const_lvalue<AcceptHandler> handler2(handler);
self_->impl_.get_service().async_accept(
self_->impl_.get_implementation(), *peer, peer_endpoint,
handler2.value, self_->impl_.get_executor());
}

private:
basic_socket_acceptor* self_;
};

class initiate_async_move_accept
{
public:
typedef Executor executor_type;

explicit initiate_async_move_accept(basic_socket_acceptor* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename MoveAcceptHandler, typename Executor1, typename Socket>
void operator()(BOOST_ASIO_MOVE_ARG(MoveAcceptHandler) handler,
const Executor1& peer_ex, endpoint_type* peer_endpoint, Socket*) const
{
BOOST_ASIO_MOVE_ACCEPT_HANDLER_CHECK(
MoveAcceptHandler, handler, Socket) type_check;

detail::non_const_lvalue<MoveAcceptHandler> handler2(handler);
self_->impl_.get_service().async_move_accept(
self_->impl_.get_implementation(), peer_ex, peer_endpoint,
handler2.value, self_->impl_.get_executor());
}

private:
basic_socket_acceptor* self_;
};

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
detail::io_object_impl<
detail::null_socket_service<Protocol>, Executor> impl_;
#elif defined(BOOST_ASIO_HAS_IOCP)
detail::io_object_impl<
detail::win_iocp_socket_service<Protocol>, Executor> impl_;
#else
detail::io_object_impl<
detail::reactive_socket_service<Protocol>, Executor> impl_;
#endif
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
