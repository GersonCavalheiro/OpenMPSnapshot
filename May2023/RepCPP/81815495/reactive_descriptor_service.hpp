
#ifndef ASIO_DETAIL_REACTIVE_DESCRIPTOR_SERVICE_HPP
#define ASIO_DETAIL_REACTIVE_DESCRIPTOR_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS) \
&& !defined(ASIO_WINDOWS_RUNTIME) \
&& !defined(__CYGWIN__)

#include "asio/associated_cancellation_slot.hpp"
#include "asio/buffer.hpp"
#include "asio/cancellation_type.hpp"
#include "asio/execution_context.hpp"
#include "asio/detail/bind_handler.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/descriptor_ops.hpp"
#include "asio/detail/descriptor_read_op.hpp"
#include "asio/detail/descriptor_write_op.hpp"
#include "asio/detail/fenced_block.hpp"
#include "asio/detail/memory.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/reactive_null_buffers_op.hpp"
#include "asio/detail/reactive_wait_op.hpp"
#include "asio/detail/reactor.hpp"
#include "asio/posix/descriptor_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class reactive_descriptor_service :
public execution_context_service_base<reactive_descriptor_service>
{
public:
typedef int native_handle_type;

class implementation_type
: private asio::detail::noncopyable
{
public:
implementation_type()
: descriptor_(-1),
state_(0)
{
}

private:
friend class reactive_descriptor_service;

int descriptor_;

descriptor_ops::state_type state_;

reactor::per_descriptor_data reactor_data_;
};

ASIO_DECL reactive_descriptor_service(execution_context& context);

ASIO_DECL void shutdown();

ASIO_DECL void construct(implementation_type& impl);

ASIO_DECL void move_construct(implementation_type& impl,
implementation_type& other_impl) ASIO_NOEXCEPT;

ASIO_DECL void move_assign(implementation_type& impl,
reactive_descriptor_service& other_service,
implementation_type& other_impl);

ASIO_DECL void destroy(implementation_type& impl);

ASIO_DECL asio::error_code assign(implementation_type& impl,
const native_handle_type& native_descriptor,
asio::error_code& ec);

bool is_open(const implementation_type& impl) const
{
return impl.descriptor_ != -1;
}

ASIO_DECL asio::error_code close(implementation_type& impl,
asio::error_code& ec);

native_handle_type native_handle(const implementation_type& impl) const
{
return impl.descriptor_;
}

ASIO_DECL native_handle_type release(implementation_type& impl);

ASIO_DECL asio::error_code cancel(implementation_type& impl,
asio::error_code& ec);

template <typename IO_Control_Command>
asio::error_code io_control(implementation_type& impl,
IO_Control_Command& command, asio::error_code& ec)
{
descriptor_ops::ioctl(impl.descriptor_, impl.state_,
command.name(), static_cast<ioctl_arg_type*>(command.data()), ec);
return ec;
}

bool non_blocking(const implementation_type& impl) const
{
return (impl.state_ & descriptor_ops::user_set_non_blocking) != 0;
}

asio::error_code non_blocking(implementation_type& impl,
bool mode, asio::error_code& ec)
{
descriptor_ops::set_user_non_blocking(
impl.descriptor_, impl.state_, mode, ec);
return ec;
}

bool native_non_blocking(const implementation_type& impl) const
{
return (impl.state_ & descriptor_ops::internal_non_blocking) != 0;
}

asio::error_code native_non_blocking(implementation_type& impl,
bool mode, asio::error_code& ec)
{
descriptor_ops::set_internal_non_blocking(
impl.descriptor_, impl.state_, mode, ec);
return ec;
}

asio::error_code wait(implementation_type& impl,
posix::descriptor_base::wait_type w, asio::error_code& ec)
{
switch (w)
{
case posix::descriptor_base::wait_read:
descriptor_ops::poll_read(impl.descriptor_, impl.state_, ec);
break;
case posix::descriptor_base::wait_write:
descriptor_ops::poll_write(impl.descriptor_, impl.state_, ec);
break;
case posix::descriptor_base::wait_error:
descriptor_ops::poll_error(impl.descriptor_, impl.state_, ec);
break;
default:
ec = asio::error::invalid_argument;
break;
}

return ec;
}

template <typename Handler, typename IoExecutor>
void async_wait(implementation_type& impl,
posix::descriptor_base::wait_type w,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef reactive_wait_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "descriptor",
&impl, impl.descriptor_, "async_wait"));

int op_type;
switch (w)
{
case posix::descriptor_base::wait_read:
op_type = reactor::read_op;
break;
case posix::descriptor_base::wait_write:
op_type = reactor::write_op;
break;
case posix::descriptor_base::wait_error:
op_type = reactor::except_op;
break;
default:
p.p->ec_ = asio::error::invalid_argument;
reactor_.post_immediate_completion(p.p, is_continuation);
p.v = p.p = 0;
return;
}

if (slot.is_connected())
{
p.p->cancellation_key_ =
&slot.template emplace<reactor_op_cancellation>(
&reactor_, &impl.reactor_data_, impl.descriptor_, op_type);
}

start_op(impl, op_type, p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename ConstBufferSequence>
size_t write_some(implementation_type& impl,
const ConstBufferSequence& buffers, asio::error_code& ec)
{
typedef buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence> bufs_type;

if (bufs_type::is_single_buffer)
{
return descriptor_ops::sync_write1(impl.descriptor_,
impl.state_, bufs_type::first(buffers).data(),
bufs_type::first(buffers).size(), ec);
}
else
{
bufs_type bufs(buffers);

return descriptor_ops::sync_write(impl.descriptor_, impl.state_,
bufs.buffers(), bufs.count(), bufs.all_empty(), ec);
}
}

size_t write_some(implementation_type& impl,
const null_buffers&, asio::error_code& ec)
{
descriptor_ops::poll_write(impl.descriptor_, impl.state_, ec);

return 0;
}

template <typename ConstBufferSequence, typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const ConstBufferSequence& buffers, Handler& handler,
const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef descriptor_write_op<ConstBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.descriptor_, buffers, handler, io_ex);

if (slot.is_connected())
{
p.p->cancellation_key_ =
&slot.template emplace<reactor_op_cancellation>(
&reactor_, &impl.reactor_data_,
impl.descriptor_, reactor::write_op);
}

ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "descriptor",
&impl, impl.descriptor_, "async_write_some"));

start_op(impl, reactor::write_op, p.p, is_continuation, true,
buffer_sequence_adapter<asio::const_buffer,
ConstBufferSequence>::all_empty(buffers));
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_write_some(implementation_type& impl,
const null_buffers&, Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

if (slot.is_connected())
{
p.p->cancellation_key_ =
&slot.template emplace<reactor_op_cancellation>(
&reactor_, &impl.reactor_data_,
impl.descriptor_, reactor::write_op);
}

ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "descriptor",
&impl, impl.descriptor_, "async_write_some(null_buffers)"));

start_op(impl, reactor::write_op, p.p, is_continuation, false, false);
p.v = p.p = 0;
}

template <typename MutableBufferSequence>
size_t read_some(implementation_type& impl,
const MutableBufferSequence& buffers, asio::error_code& ec)
{
typedef buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence> bufs_type;

if (bufs_type::is_single_buffer)
{
return descriptor_ops::sync_read1(impl.descriptor_,
impl.state_, bufs_type::first(buffers).data(),
bufs_type::first(buffers).size(), ec);
}
else
{
bufs_type bufs(buffers);

return descriptor_ops::sync_read(impl.descriptor_, impl.state_,
bufs.buffers(), bufs.count(), bufs.all_empty(), ec);
}
}

size_t read_some(implementation_type& impl,
const null_buffers&, asio::error_code& ec)
{
descriptor_ops::poll_read(impl.descriptor_, impl.state_, ec);

return 0;
}

template <typename MutableBufferSequence,
typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const MutableBufferSequence& buffers,
Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef descriptor_read_op<MutableBufferSequence, Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, impl.descriptor_, buffers, handler, io_ex);

if (slot.is_connected())
{
p.p->cancellation_key_ =
&slot.template emplace<reactor_op_cancellation>(
&reactor_, &impl.reactor_data_,
impl.descriptor_, reactor::read_op);
}

ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "descriptor",
&impl, impl.descriptor_, "async_read_some"));

start_op(impl, reactor::read_op, p.p, is_continuation, true,
buffer_sequence_adapter<asio::mutable_buffer,
MutableBufferSequence>::all_empty(buffers));
p.v = p.p = 0;
}

template <typename Handler, typename IoExecutor>
void async_read_some(implementation_type& impl,
const null_buffers&, Handler& handler, const IoExecutor& io_ex)
{
bool is_continuation =
asio_handler_cont_helpers::is_continuation(handler);

typename associated_cancellation_slot<Handler>::type slot
= asio::get_associated_cancellation_slot(handler);

typedef reactive_null_buffers_op<Handler, IoExecutor> op;
typename op::ptr p = { asio::detail::addressof(handler),
op::ptr::allocate(handler), 0 };
p.p = new (p.v) op(success_ec_, handler, io_ex);

if (slot.is_connected())
{
p.p->cancellation_key_ =
&slot.template emplace<reactor_op_cancellation>(
&reactor_, &impl.reactor_data_,
impl.descriptor_, reactor::read_op);
}

ASIO_HANDLER_CREATION((reactor_.context(), *p.p, "descriptor",
&impl, impl.descriptor_, "async_read_some(null_buffers)"));

start_op(impl, reactor::read_op, p.p, is_continuation, false, false);
p.v = p.p = 0;
}

private:
ASIO_DECL void start_op(implementation_type& impl, int op_type,
reactor_op* op, bool is_continuation, bool is_non_blocking, bool noop);

class reactor_op_cancellation
{
public:
reactor_op_cancellation(reactor* r,
reactor::per_descriptor_data* p, int d, int o)
: reactor_(r),
reactor_data_(p),
descriptor_(d),
op_type_(o)
{
}

void operator()(cancellation_type_t type)
{
if (!!(type &
(cancellation_type::terminal
| cancellation_type::partial
| cancellation_type::total)))
{
reactor_->cancel_ops_by_key(descriptor_,
*reactor_data_, op_type_, this);
}
}

private:
reactor* reactor_;
reactor::per_descriptor_data* reactor_data_;
int descriptor_;
int op_type_;
};

reactor& reactor_;

const asio::error_code success_ec_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/reactive_descriptor_service.ipp"
#endif 

#endif 

#endif 
