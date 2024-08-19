
#ifndef BOOST_ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP
#define BOOST_ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_IOCP)

#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/detail/buffer_sequence_adapter.hpp>
#include <boost/asio/detail/cstdint.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/mutex.hpp>
#include <boost/asio/detail/operation.hpp>
#include <boost/asio/detail/win_iocp_handle_read_op.hpp>
#include <boost/asio/detail/win_iocp_handle_write_op.hpp>
#include <boost/asio/detail/win_iocp_io_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class win_iocp_handle_service :
public execution_context_service_base<win_iocp_handle_service>
{
public:
typedef HANDLE native_handle_type;

class implementation_type
{
public:
implementation_type()
: handle_(INVALID_HANDLE_VALUE),
safe_cancellation_thread_id_(0),
next_(0),
prev_(0)
{
}

private:
friend class win_iocp_handle_service;

native_handle_type handle_;

DWORD safe_cancellation_thread_id_;

implementation_type* next_;
implementation_type* prev_;
};

BOOST_ASIO_DECL win_iocp_handle_service(execution_context& context);

BOOST_ASIO_DECL void shutdown();

BOOST_ASIO_DECL void construct(implementation_type& impl);

BOOST_ASIO_DECL void move_construct(implementation_type& impl,
implementation_type& other_impl);

BOOST_ASIO_DECL void move_assign(implementation_type& impl,
win_iocp_handle_service& other_service,
implementation_type& other_impl);

BOOST_ASIO_DECL void destroy(implementation_type& impl);

BOOST_ASIO_DECL boost::system::error_code assign(implementation_type& impl,
const native_handle_type& handle, boost::system::error_code& ec);

bool is_open(const implementation_type& impl) const
{
return impl.handle_ != INVALID_HANDLE_VALUE;
}

BOOST_ASIO_DECL boost::system::error_code close(implementation_type& impl,
boost::system::error_code& ec);

native_handle_type native_handle(const implementation_type& impl) const
{
return impl.handle_;
}

BOOST_ASIO_DECL boost::system::error_code cancel(implementation_type& impl,
boost::system::error_code& ec);

template <typename ConstBufferSequence>
size_t write_some(implementation_type& impl,
const ConstBufferSequence& buffers, boost::system::error_code& ec)
{
return write_some_at(impl, 0, buffers, ec);
}

template <typename ConstBufferSequence>
size_t write_some_at(implementation_type& impl, uint64_t offset,
const ConstBufferSequence& buffers, boost::system::error_code& ec)
{
boost::asio::const_buffer buffer =
buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::first(buffers);

return do_write(impl, offset, buffer, ec);
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const ConstBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_handle_write_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_write_some"));

start_write_op(impl, 0,
buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::first(buffers), p.p);
p.v = p.p = 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some_at(implementation_type& impl,
uint64_t offset, const ConstBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_handle_write_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_write_some_at"));

start_write_op(impl, offset,
buffer_sequence_adapter<boost::asio::const_buffer,
ConstBufferSequence>::first(buffers), p.p);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t read_some(implementation_type& impl,
const MutableBufferSequence& buffers, boost::system::error_code& ec)
{
return read_some_at(impl, 0, buffers, ec);
}

template <typename MutableBufferSequence>
size_t read_some_at(implementation_type& impl, uint64_t offset,
const MutableBufferSequence& buffers, boost::system::error_code& ec)
{
boost::asio::mutable_buffer buffer =
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::first(buffers);

return do_read(impl, offset, buffer, ec);
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_handle_read_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_read_some"));

start_read_op(impl, 0,
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::first(buffers), p.p);
p.v = p.p = 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some_at(implementation_type& impl,
uint64_t offset, const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typedef win_iocp_handle_read_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { boost::asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(buffers, handler, io_ex);

BOOST_ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_read_some_at"));

start_read_op(impl, offset,
buffer_sequence_adapter<boost::asio::mutable_buffer,
MutableBufferSequence>::first(buffers), p.p);
p.v = p.p = 0;
}

private:
size_t write_some(implementation_type& impl,
const null_buffers& buffers, boost::system::error_code& ec);
size_t write_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, boost::system::error_code& ec);
template <typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const null_buffers& buffers, Handler& handler,
const IoExecutor& io_ex);
template <typename Handler, typename IoExecutor>
void async_write_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, Handler& handler, const IoExecutor& io_ex);
size_t read_some(implementation_type& impl,
const null_buffers& buffers, boost::system::error_code& ec);
size_t read_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, boost::system::error_code& ec);
template <typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const null_buffers& buffers, Handler& handler,
const IoExecutor& io_ex);
template <typename Handler, typename IoExecutor>
void async_read_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, Handler& handler, const IoExecutor& io_ex);

class overlapped_wrapper;

BOOST_ASIO_DECL size_t do_write(implementation_type& impl,
uint64_t offset, const boost::asio::const_buffer& buffer,
boost::system::error_code& ec);

BOOST_ASIO_DECL void start_write_op(implementation_type& impl,
uint64_t offset, const boost::asio::const_buffer& buffer,
operation* op);

BOOST_ASIO_DECL size_t do_read(implementation_type& impl,
uint64_t offset, const boost::asio::mutable_buffer& buffer,
boost::system::error_code& ec);

BOOST_ASIO_DECL void start_read_op(implementation_type& impl,
uint64_t offset, const boost::asio::mutable_buffer& buffer,
operation* op);

BOOST_ASIO_DECL void update_cancellation_thread_id(implementation_type& impl);

BOOST_ASIO_DECL void close_for_destruction(implementation_type& impl);

win_iocp_io_context& iocp_service_;

mutex mutex_;

implementation_type* impl_list_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_iocp_handle_service.ipp>
#endif 

#endif 

#endif 
