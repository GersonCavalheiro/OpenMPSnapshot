
#ifndef ASIO_SSL_DETAIL_IO_HPP
#define ASIO_SSL_DETAIL_IO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/base_from_cancellation_state.hpp"
#include "asio/detail/handler_tracking.hpp"
#include "asio/ssl/detail/engine.hpp"
#include "asio/ssl/detail/stream_core.hpp"
#include "asio/write.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

template <typename Stream, typename Operation>
std::size_t io(Stream& next_layer, stream_core& core,
const Operation& op, asio::error_code& ec)
{
asio::error_code io_ec;
std::size_t bytes_transferred = 0;
do switch (op(core.engine_, ec, bytes_transferred))
{
case engine::want_input_and_retry:

if (core.input_.size() == 0)
{
core.input_ = asio::buffer(core.input_buffer_,
next_layer.read_some(core.input_buffer_, io_ec));
if (!ec)
ec = io_ec;
}

core.input_ = core.engine_.put_input(core.input_);

continue;

case engine::want_output_and_retry:

asio::write(next_layer,
core.engine_.get_output(core.output_buffer_), io_ec);
if (!ec)
ec = io_ec;

continue;

case engine::want_output:

asio::write(next_layer,
core.engine_.get_output(core.output_buffer_), io_ec);
if (!ec)
ec = io_ec;

core.engine_.map_error_code(ec);
return bytes_transferred;

default:

core.engine_.map_error_code(ec);
return bytes_transferred;

} while (!ec);

core.engine_.map_error_code(ec);
return 0;
}

template <typename Stream, typename Operation, typename Handler>
class io_op
: public asio::detail::base_from_cancellation_state<Handler>
{
public:
io_op(Stream& next_layer, stream_core& core,
const Operation& op, Handler& handler)
: asio::detail::base_from_cancellation_state<Handler>(handler),
next_layer_(next_layer),
core_(core),
op_(op),
start_(0),
want_(engine::want_nothing),
bytes_transferred_(0),
handler_(ASIO_MOVE_CAST(Handler)(handler))
{
}

#if defined(ASIO_HAS_MOVE)
io_op(const io_op& other)
: asio::detail::base_from_cancellation_state<Handler>(other),
next_layer_(other.next_layer_),
core_(other.core_),
op_(other.op_),
start_(other.start_),
want_(other.want_),
ec_(other.ec_),
bytes_transferred_(other.bytes_transferred_),
handler_(other.handler_)
{
}

io_op(io_op&& other)
: asio::detail::base_from_cancellation_state<Handler>(
ASIO_MOVE_CAST(
asio::detail::base_from_cancellation_state<Handler>)(
other)),
next_layer_(other.next_layer_),
core_(other.core_),
op_(ASIO_MOVE_CAST(Operation)(other.op_)),
start_(other.start_),
want_(other.want_),
ec_(other.ec_),
bytes_transferred_(other.bytes_transferred_),
handler_(ASIO_MOVE_CAST(Handler)(other.handler_))
{
}
#endif 

void operator()(asio::error_code ec,
std::size_t bytes_transferred = ~std::size_t(0), int start = 0)
{
switch (start_ = start)
{
case 1: 
do
{
switch (want_ = op_(core_.engine_, ec_, bytes_transferred_))
{
case engine::want_input_and_retry:

if (core_.input_.size() != 0)
{
core_.input_ = core_.engine_.put_input(core_.input_);
continue;
}

if (core_.expiry(core_.pending_read_) == core_.neg_infin())
{
core_.pending_read_.expires_at(core_.pos_infin());

ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, Operation::tracking_name()));

next_layer_.async_read_some(
asio::buffer(core_.input_buffer_),
ASIO_MOVE_CAST(io_op)(*this));
}
else
{
ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, Operation::tracking_name()));

core_.pending_read_.async_wait(ASIO_MOVE_CAST(io_op)(*this));
}

return;

case engine::want_output_and_retry:
case engine::want_output:

if (core_.expiry(core_.pending_write_) == core_.neg_infin())
{
core_.pending_write_.expires_at(core_.pos_infin());

ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, Operation::tracking_name()));

asio::async_write(next_layer_,
core_.engine_.get_output(core_.output_buffer_),
ASIO_MOVE_CAST(io_op)(*this));
}
else
{
ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, Operation::tracking_name()));

core_.pending_write_.async_wait(ASIO_MOVE_CAST(io_op)(*this));
}

return;

default:

if (start)
{
ASIO_HANDLER_LOCATION((
__FILE__, __LINE__, Operation::tracking_name()));

next_layer_.async_read_some(
asio::buffer(core_.input_buffer_, 0),
ASIO_MOVE_CAST(io_op)(*this));

return;
}
else
{
break;
}
}

default:
if (bytes_transferred == ~std::size_t(0))
bytes_transferred = 0; 
else if (!ec_)
ec_ = ec;

switch (want_)
{
case engine::want_input_and_retry:

core_.input_ = asio::buffer(
core_.input_buffer_, bytes_transferred);
core_.input_ = core_.engine_.put_input(core_.input_);

core_.pending_read_.expires_at(core_.neg_infin());

if (this->cancelled() != cancellation_type::none)
{
ec_ = asio::error::operation_aborted;
break;
}

continue;

case engine::want_output_and_retry:

core_.pending_write_.expires_at(core_.neg_infin());

if (this->cancelled() != cancellation_type::none)
{
ec_ = asio::error::operation_aborted;
break;
}

continue;

case engine::want_output:

core_.pending_write_.expires_at(core_.neg_infin());


default:

op_.call_handler(handler_,
core_.engine_.map_error_code(ec_),
ec_ ? 0 : bytes_transferred_);

return;
}
} while (!ec_);

op_.call_handler(handler_, core_.engine_.map_error_code(ec_), 0);
}
}

Stream& next_layer_;
stream_core& core_;
Operation op_;
int start_;
engine::want want_;
asio::error_code ec_;
std::size_t bytes_transferred_;
Handler handler_;
};

template <typename Stream, typename Operation, typename Handler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
io_op<Stream, Operation, Handler>* this_handler)
{
#if defined(ASIO_NO_DEPRECATED)
asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename Stream, typename Operation, typename Handler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
io_op<Stream, Operation, Handler>* this_handler)
{
asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename Stream, typename Operation, typename Handler>
inline bool asio_handler_is_continuation(
io_op<Stream, Operation, Handler>* this_handler)
{
return this_handler->start_ == 0 ? true
: asio_handler_cont_helpers::is_continuation(this_handler->handler_);
}

template <typename Function, typename Stream,
typename Operation, typename Handler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
io_op<Stream, Operation, Handler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename Stream,
typename Operation, typename Handler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
io_op<Stream, Operation, Handler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Stream, typename Operation, typename Handler>
inline void async_io(Stream& next_layer, stream_core& core,
const Operation& op, Handler& handler)
{
io_op<Stream, Operation, Handler>(
next_layer, core, op, handler)(
asio::error_code(), 0, 1);
}

} 
} 

template <template <typename, typename> class Associator,
typename Stream, typename Operation,
typename Handler, typename DefaultCandidate>
struct associator<Associator,
ssl::detail::io_op<Stream, Operation, Handler>,
DefaultCandidate>
: Associator<Handler, DefaultCandidate>
{
static typename Associator<Handler, DefaultCandidate>::type get(
const ssl::detail::io_op<Stream, Operation, Handler>& h,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<Handler, DefaultCandidate>::get(h.handler_, c);
}
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
