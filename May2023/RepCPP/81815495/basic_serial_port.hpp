
#ifndef ASIO_BASIC_SERIAL_PORT_HPP
#define ASIO_BASIC_SERIAL_PORT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_SERIAL_PORT) \
|| defined(GENERATING_DOCUMENTATION)

#include <string>
#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/non_const_lvalue.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/serial_port_base.hpp"
#if defined(ASIO_HAS_IOCP)
# include "asio/detail/win_iocp_serial_port_service.hpp"
#else
# include "asio/detail/reactive_serial_port_service.hpp"
#endif

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {


template <typename Executor = any_io_executor>
class basic_serial_port
: public serial_port_base
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_serial_port<Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#elif defined(ASIO_HAS_IOCP)
typedef detail::win_iocp_serial_port_service::native_handle_type
native_handle_type;
#else
typedef detail::reactive_serial_port_service::native_handle_type
native_handle_type;
#endif

typedef basic_serial_port lowest_layer_type;


explicit basic_serial_port(const executor_type& ex)
: impl_(0, ex)
{
}


template <typename ExecutionContext>
explicit basic_serial_port(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
}


basic_serial_port(const executor_type& ex, const char* device)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), device, ec);
asio::detail::throw_error(ec, "open");
}


template <typename ExecutionContext>
basic_serial_port(ExecutionContext& context, const char* device,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), device, ec);
asio::detail::throw_error(ec, "open");
}


basic_serial_port(const executor_type& ex, const std::string& device)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), device, ec);
asio::detail::throw_error(ec, "open");
}


template <typename ExecutionContext>
basic_serial_port(ExecutionContext& context, const std::string& device,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), device, ec);
asio::detail::throw_error(ec, "open");
}


basic_serial_port(const executor_type& ex,
const native_handle_type& native_serial_port)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_serial_port, ec);
asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_serial_port(ExecutionContext& context,
const native_handle_type& native_serial_port,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_serial_port, ec);
asio::detail::throw_error(ec, "assign");
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_serial_port(basic_serial_port&& other)
: impl_(std::move(other.impl_))
{
}


basic_serial_port& operator=(basic_serial_port&& other)
{
impl_ = std::move(other.impl_);
return *this;
}
#endif 


~basic_serial_port()
{
}

executor_type get_executor() ASIO_NOEXCEPT
{
return impl_.get_executor();
}


lowest_layer_type& lowest_layer()
{
return *this;
}


const lowest_layer_type& lowest_layer() const
{
return *this;
}


void open(const std::string& device)
{
asio::error_code ec;
impl_.get_service().open(impl_.get_implementation(), device, ec);
asio::detail::throw_error(ec, "open");
}


ASIO_SYNC_OP_VOID open(const std::string& device,
asio::error_code& ec)
{
impl_.get_service().open(impl_.get_implementation(), device, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void assign(const native_handle_type& native_serial_port)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_serial_port, ec);
asio::detail::throw_error(ec, "assign");
}


ASIO_SYNC_OP_VOID assign(const native_handle_type& native_serial_port,
asio::error_code& ec)
{
impl_.get_service().assign(impl_.get_implementation(),
native_serial_port, ec);
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


native_handle_type native_handle()
{
return impl_.get_service().native_handle(impl_.get_implementation());
}


void cancel()
{
asio::error_code ec;
impl_.get_service().cancel(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "cancel");
}


ASIO_SYNC_OP_VOID cancel(asio::error_code& ec)
{
impl_.get_service().cancel(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void send_break()
{
asio::error_code ec;
impl_.get_service().send_break(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "send_break");
}


ASIO_SYNC_OP_VOID send_break(asio::error_code& ec)
{
impl_.get_service().send_break(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename SettableSerialPortOption>
void set_option(const SettableSerialPortOption& option)
{
asio::error_code ec;
impl_.get_service().set_option(impl_.get_implementation(), option, ec);
asio::detail::throw_error(ec, "set_option");
}


template <typename SettableSerialPortOption>
ASIO_SYNC_OP_VOID set_option(const SettableSerialPortOption& option,
asio::error_code& ec)
{
impl_.get_service().set_option(impl_.get_implementation(), option, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename GettableSerialPortOption>
void get_option(GettableSerialPortOption& option) const
{
asio::error_code ec;
impl_.get_service().get_option(impl_.get_implementation(), option, ec);
asio::detail::throw_error(ec, "get_option");
}


template <typename GettableSerialPortOption>
ASIO_SYNC_OP_VOID get_option(GettableSerialPortOption& option,
asio::error_code& ec) const
{
impl_.get_service().get_option(impl_.get_implementation(), option, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers)
{
asio::error_code ec;
std::size_t s = impl_.get_service().write_some(
impl_.get_implementation(), buffers, ec);
asio::detail::throw_error(ec, "write_some");
return s;
}


template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers,
asio::error_code& ec)
{
return impl_.get_service().write_some(
impl_.get_implementation(), buffers, ec);
}


template <typename ConstBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) WriteHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (asio::error_code, std::size_t))
async_write_some(const ConstBufferSequence& buffers,
ASIO_MOVE_ARG(WriteHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WriteHandler,
void (asio::error_code, std::size_t)>(
initiate_async_write_some(this), handler, buffers);
}


template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers)
{
asio::error_code ec;
std::size_t s = impl_.get_service().read_some(
impl_.get_implementation(), buffers, ec);
asio::detail::throw_error(ec, "read_some");
return s;
}


template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers,
asio::error_code& ec)
{
return impl_.get_service().read_some(
impl_.get_implementation(), buffers, ec);
}


template <typename MutableBufferSequence,
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
async_read_some(const MutableBufferSequence& buffers,
ASIO_MOVE_ARG(ReadHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<ReadHandler,
void (asio::error_code, std::size_t)>(
initiate_async_read_some(this), handler, buffers);
}

private:
basic_serial_port(const basic_serial_port&) ASIO_DELETED;
basic_serial_port& operator=(const basic_serial_port&) ASIO_DELETED;

class initiate_async_write_some
{
public:
typedef Executor executor_type;

explicit initiate_async_write_some(basic_serial_port* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WriteHandler, typename ConstBufferSequence>
void operator()(ASIO_MOVE_ARG(WriteHandler) handler,
const ConstBufferSequence& buffers) const
{
ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

detail::non_const_lvalue<WriteHandler> handler2(handler);
self_->impl_.get_service().async_write_some(
self_->impl_.get_implementation(), buffers,
handler2.value, self_->impl_.get_executor());
}

private:
basic_serial_port* self_;
};

class initiate_async_read_some
{
public:
typedef Executor executor_type;

explicit initiate_async_read_some(basic_serial_port* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename ReadHandler, typename MutableBufferSequence>
void operator()(ASIO_MOVE_ARG(ReadHandler) handler,
const MutableBufferSequence& buffers) const
{
ASIO_READ_HANDLER_CHECK(ReadHandler, handler) type_check;

detail::non_const_lvalue<ReadHandler> handler2(handler);
self_->impl_.get_service().async_read_some(
self_->impl_.get_implementation(), buffers,
handler2.value, self_->impl_.get_executor());
}

private:
basic_serial_port* self_;
};

#if defined(ASIO_HAS_IOCP)
detail::io_object_impl<detail::win_iocp_serial_port_service, Executor> impl_;
#else
detail::io_object_impl<detail::reactive_serial_port_service, Executor> impl_;
#endif
};

} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
