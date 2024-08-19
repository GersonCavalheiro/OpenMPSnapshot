
#ifndef ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP
#define ASIO_DETAIL_WIN_IOCP_HANDLE_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/associated_cancellation_slot.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/cstdint.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/mutex.hpp"
#include "asio/detail/operation.hpp"
#include "asio/detail/win_iocp_handle_read_op.hpp"
#include "asio/detail/win_iocp_handle_write_op.hpp"
#include "asio/detail/win_iocp_io_context.hpp"

#include "asio/detail/push_options.hpp"

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

ASIO_DECL win_iocp_handle_service(execution_context& context);

ASIO_DECL void shutdown();

ASIO_DECL void construct(implementation_type& impl);

ASIO_DECL void move_construct(implementation_type& impl,
implementation_type& other_impl);

ASIO_DECL void move_assign(implementation_type& impl,
win_iocp_handle_service& other_service,
implementation_type& other_impl);

ASIO_DECL void destroy(implementation_type& impl);

ASIO_DECL asio::error_code assign(implementation_type& impl,
const native_handle_type& handle, asio::error_code& ec);

bool is_open(const implementation_type& impl) const
{
return impl.handle_ != INVALID_HANDLE_VALUE;
}

ASIO_DECL asio::error_code close(implementation_type& impl,
asio::error_code& ec);

native_handle_type native_handle(const implementation_type& impl) const
{
return impl.handle_;
}

ASIO_DECL asio::error_code cancel(implementation_type& impl,
asio::error_code& ec);

template <typename ConstBufferSequence>
size_t write_some(implementation_type& impl,
const ConstBufferSequence& buffers, asio::error_code& ec)
{
return write_some_at(impl, 0, buffers, ec);
}

template <typename ConstBufferSequence>
size_t write_some_at(implementation_type& impl, uint64_t offset,
const ConstBufferSequence& buffers, asio::error_code& ec)
{
asio::const_buffer buffer =
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::first(buffers);

return do_write(impl, offset, buffer, ec);
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const ConstBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_handle_write_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(buffers, handler, io_ex);

ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_write_some"));

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.handle_, o);

start_write_op(impl, 0,
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::first(buffers), o);
p.v = p.p = 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some_at(implementation_type& impl,
uint64_t offset, const ConstBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_handle_write_op<
ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(buffers, handler, io_ex);

ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_write_some_at"));

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.handle_, o);

start_write_op(impl, offset,
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::first(buffers), o);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t read_some(implementation_type& impl,
const MutableBufferSequence& buffers, asio::error_code& ec)
{
return read_some_at(impl, 0, buffers, ec);
}

template <typename MutableBufferSequence>
size_t read_some_at(implementation_type& impl, uint64_t offset,
const MutableBufferSequence& buffers, asio::error_code& ec)
{
asio::mutable_buffer buffer =
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::first(buffers);

return do_read(impl, offset, buffer, ec);
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_handle_read_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(buffers, handler, io_ex);

ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_read_some"));

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.handle_, o);

start_read_op(impl, 0,
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::first(buffers), o);
p.v = p.p = 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some_at(implementation_type& impl,
uint64_t offset, const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef win_iocp_handle_read_op<
MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
operation* o = p.p = new (p.v) op(buffers, handler, io_ex);

ASIO_HANDLER_CREATION((iocp_service_.context(), *p.p, "handle", &impl,
reinterpret_cast<uintmax_t>(impl.handle_), "async_read_some_at"));

if (slot.is_connected())
o = &slot.template emplace<iocp_op_cancellation>(impl.handle_, o);

start_read_op(impl, offset,
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::first(buffers), o);
p.v = p.p = 0;
}

private:
size_t write_some(implementation_type& impl,
const null_buffers& buffers, asio::error_code& ec);
size_t write_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, asio::error_code& ec);
template <typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const null_buffers& buffers, Handler& handler,
const IoExecutor& io_ex);
template <typename Handler, typename IoExecutor>
void async_write_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, Handler& handler, const IoExecutor& io_ex);
size_t read_some(implementation_type& impl,
const null_buffers& buffers, asio::error_code& ec);
size_t read_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, asio::error_code& ec);
template <typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const null_buffers& buffers, Handler& handler,
const IoExecutor& io_ex);
template <typename Handler, typename IoExecutor>
void async_read_some_at(implementation_type& impl, uint64_t offset,
const null_buffers& buffers, Handler& handler, const IoExecutor& io_ex);

class overlapped_wrapper;

ASIO_DECL size_t do_write(implementation_type& impl,
uint64_t offset, const asio::const_buffer& buffer,
asio::error_code& ec);

ASIO_DECL void start_write_op(implementation_type& impl,
uint64_t offset, const asio::const_buffer& buffer,
operation* op);

ASIO_DECL size_t do_read(implementation_type& impl,
uint64_t offset, const asio::mutable_buffer& buffer,
asio::error_code& ec);

ASIO_DECL void start_read_op(implementation_type& impl,
uint64_t offset, const asio::mutable_buffer& buffer,
operation* op);

ASIO_DECL void update_cancellation_thread_id(implementation_type& impl);

ASIO_DECL void close_for_destruction(implementation_type& impl);

class iocp_op_cancellation : public operation
{
public:
iocp_op_cancellation(HANDLE h, operation* target)
: operation(&iocp_op_cancellation::do_complete),
handle_(h),
target_(target)
{
}

static void do_complete(void* owner, operation* base,
const asio::error_code& result_ec,
std::size_t bytes_transferred)
{
iocp_op_cancellation* o = static_cast<iocp_op_cancellation*>(base);
o->target_->complete(owner, result_ec, bytes_transferred);
}

void operator()(cancellation_type_t type)
{
#if defined(_WIN32_WINNT) && (_WIN32_WINNT >= 0x0600)
if (!!(type &
(cancellation_type::terminal
| cancellation_type::partial
| cancellation_type::total)))
{
::CancelIoEx(handle_, this);
}
#else 
(void)type;
#endif 
}

private:
HANDLE handle_;
operation* target_;
};

win_iocp_io_context& iocp_service_;

mutex mutex_;

implementation_type* impl_list_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_iocp_handle_service.ipp"
#endif 

#endif 

#endif 
