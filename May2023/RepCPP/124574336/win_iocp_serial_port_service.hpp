
#ifndef BOOST_ASIO_DETAIL_WIN_IOCP_SERIAL_PORT_SERVICE_HPP
#define BOOST_ASIO_DETAIL_WIN_IOCP_SERIAL_PORT_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP) && defined(BOOST_ASIO_HAS_SERIAL_PORT)

#include <string>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/detail/win_iocp_handle_service.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class win_iocp_serial_port_service :
public execution_context_service_base<win_iocp_serial_port_service>
{
public:
typedef win_iocp_handle_service::native_handle_type native_handle_type;

typedef win_iocp_handle_service::implementation_type implementation_type;

BOOST_ASIO_DECL win_iocp_serial_port_service(execution_context& context);

BOOST_ASIO_DECL void shutdown();

void construct(implementation_type& impl)
{
handle_service_.construct(impl);
}

void move_construct(implementation_type& impl,
implementation_type& other_impl)
{
handle_service_.move_construct(impl, other_impl);
}

void move_assign(implementation_type& impl,
win_iocp_serial_port_service& other_service,
implementation_type& other_impl)
{
handle_service_.move_assign(impl,
other_service.handle_service_, other_impl);
}

void destroy(implementation_type& impl)
{
handle_service_.destroy(impl);
}

BOOST_ASIO_DECL boost::system::error_code open(implementation_type& impl,
const std::string& device, boost::system::error_code& ec);

boost::system::error_code assign(implementation_type& impl,
const native_handle_type& handle, boost::system::error_code& ec)
{
return handle_service_.assign(impl, handle, ec);
}

bool is_open(const implementation_type& impl) const
{
return handle_service_.is_open(impl);
}

boost::system::error_code close(implementation_type& impl,
boost::system::error_code& ec)
{
return handle_service_.close(impl, ec);
}

native_handle_type native_handle(implementation_type& impl)
{
return handle_service_.native_handle(impl);
}

boost::system::error_code cancel(implementation_type& impl,
boost::system::error_code& ec)
{
return handle_service_.cancel(impl, ec);
}

template <typename SettableSerialPortOption>
boost::system::error_code set_option(implementation_type& impl,
const SettableSerialPortOption& option, boost::system::error_code& ec)
{
return do_set_option(impl,
&win_iocp_serial_port_service::store_option<SettableSerialPortOption>,
&option, ec);
}

template <typename GettableSerialPortOption>
boost::system::error_code get_option(const implementation_type& impl,
GettableSerialPortOption& option, boost::system::error_code& ec) const
{
return do_get_option(impl,
&win_iocp_serial_port_service::load_option<GettableSerialPortOption>,
&option, ec);
}

boost::system::error_code send_break(implementation_type&,
boost::system::error_code& ec)
{
ec = boost::asio::error::operation_not_supported;
return ec;
}

template <typename ConstBufferSequence>
size_t write_some(implementation_type& impl,
const ConstBufferSequence& buffers, boost::system::error_code& ec)
{
return handle_service_.write_some(impl, buffers, ec);
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const ConstBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
handle_service_.async_write_some(impl, buffers, handler, io_ex);
}

template <typename MutableBufferSequence>
size_t read_some(implementation_type& impl,
const MutableBufferSequence& buffers, boost::system::error_code& ec)
{
return handle_service_.read_some(impl, buffers, ec);
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
handle_service_.async_read_some(impl, buffers, handler, io_ex);
}

private:
typedef boost::system::error_code (*store_function_type)(
const void*, ::DCB&, boost::system::error_code&);

template <typename SettableSerialPortOption>
static boost::system::error_code store_option(const void* option,
::DCB& storage, boost::system::error_code& ec)
{
static_cast<const SettableSerialPortOption*>(option)->store(storage, ec);
return ec;
}

BOOST_ASIO_DECL boost::system::error_code do_set_option(
implementation_type& impl, store_function_type store,
const void* option, boost::system::error_code& ec);

typedef boost::system::error_code (*load_function_type)(
void*, const ::DCB&, boost::system::error_code&);

template <typename GettableSerialPortOption>
static boost::system::error_code load_option(void* option,
const ::DCB& storage, boost::system::error_code& ec)
{
static_cast<GettableSerialPortOption*>(option)->load(storage, ec);
return ec;
}

BOOST_ASIO_DECL boost::system::error_code do_get_option(
const implementation_type& impl, load_function_type load,
void* option, boost::system::error_code& ec) const;

win_iocp_handle_service handle_service_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_iocp_serial_port_service.ipp>
#endif 

#endif 

#endif 
