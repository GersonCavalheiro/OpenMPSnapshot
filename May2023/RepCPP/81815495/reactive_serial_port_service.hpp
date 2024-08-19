
#ifndef ASIO_DETAIL_REACTIVE_SERIAL_PORT_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_SERIAL_PORT_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_SERIAL_PORT)
#if !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)

#include <string>
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/serial_port_base.hpp"
#include "asio/detail/descriptor_ops.hpp"
#include "asio/detail/reactive_descriptor_service.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class reactive_serial_port_service :
public execution_context_service_base<reactive_serial_port_service>
{
public:
typedef reactive_descriptor_service::native_handle_type native_handle_type;

typedef reactive_descriptor_service::implementation_type implementation_type;

ASIO_DECL reactive_serial_port_service(execution_context& context);

ASIO_DECL void shutdown();

void construct(implementation_type& impl)
{
descriptor_service_.construct(impl);
}

void move_construct(implementation_type& impl,
implementation_type& other_impl)
{
descriptor_service_.move_construct(impl, other_impl);
}

void move_assign(implementation_type& impl,
reactive_serial_port_service& other_service,
implementation_type& other_impl)
{
descriptor_service_.move_assign(impl,
other_service.descriptor_service_, other_impl);
}

void destroy(implementation_type& impl)
{
descriptor_service_.destroy(impl);
}

ASIO_DECL asio::error_code open(implementation_type& impl,
const std::string& device, asio::error_code& ec);

asio::error_code assign(implementation_type& impl,
const native_handle_type& native_descriptor,
asio::error_code& ec)
{
return descriptor_service_.assign(impl, native_descriptor, ec);
}

bool is_open(const implementation_type& impl) const
{
return descriptor_service_.is_open(impl);
}

asio::error_code close(implementation_type& impl,
asio::error_code& ec)
{
return descriptor_service_.close(impl, ec);
}

native_handle_type native_handle(implementation_type& impl)
{
return descriptor_service_.native_handle(impl);
}

asio::error_code cancel(implementation_type& impl,
asio::error_code& ec)
{
return descriptor_service_.cancel(impl, ec);
}

template <typename SettableSerialPortOption>
asio::error_code set_option(implementation_type& impl,
const SettableSerialPortOption& option, asio::error_code& ec)
{
return do_set_option(impl,
&reactive_serial_port_service::store_option<SettableSerialPortOption>,
&option, ec);
}

template <typename GettableSerialPortOption>
asio::error_code get_option(const implementation_type& impl,
GettableSerialPortOption& option, asio::error_code& ec) const
{
return do_get_option(impl,
&reactive_serial_port_service::load_option<GettableSerialPortOption>,
&option, ec);
}

asio::error_code send_break(implementation_type& impl,
asio::error_code& ec)
{
int result = ::tcsendbreak(descriptor_service_.native_handle(impl), 0);
descriptor_ops::get_last_error(ec, result < 0);
return ec;
}

template <typename ConstBufferSequence>
size_t write_some(implementation_type& impl,
const ConstBufferSequence& buffers, asio::error_code& ec)
{
return descriptor_service_.write_some(impl, buffers, ec);
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const ConstBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
descriptor_service_.async_write_some(impl, buffers, handler, io_ex);
}

template <typename MutableBufferSequence>
size_t read_some(implementation_type& impl,
const MutableBufferSequence& buffers, asio::error_code& ec)
{
return descriptor_service_.read_some(impl, buffers, ec);
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
descriptor_service_.async_read_some(impl, buffers, handler, io_ex);
}

private:
typedef asio::error_code (*store_function_type)(
const void*, termios&, asio::error_code&);

template <typename SettableSerialPortOption>
static asio::error_code store_option(const void* option,
termios& storage, asio::error_code& ec)
{
static_cast<const SettableSerialPortOption*>(option)->store(storage, ec);
return ec;
}

ASIO_DECL asio::error_code do_set_option(
implementation_type& impl, store_function_type store,
const void* option, asio::error_code& ec);

typedef asio::error_code (*load_function_type)(
void*, const termios&, asio::error_code&);

template <typename GettableSerialPortOption>
static asio::error_code load_option(void* option,
const termios& storage, asio::error_code& ec)
{
static_cast<GettableSerialPortOption*>(option)->load(storage, ec);
return ec;
}

ASIO_DECL asio::error_code do_get_option(
const implementation_type& impl, load_function_type load,
void* option, asio::error_code& ec) const;

reactive_descriptor_service descriptor_service_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/reactive_serial_port_service.ipp"
#endif 

#endif 
#endif 

#endif 
