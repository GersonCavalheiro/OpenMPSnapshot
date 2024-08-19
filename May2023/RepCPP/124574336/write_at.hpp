
#ifndef BOOST_ASIO_IMPL_WRITE_AT_HPP
#define BOOST_ASIO_IMPL_WRITE_AT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/associated_allocator.hpp>
#include <boost/asio/associated_executor.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/completion_condition.hpp>
#include <boost/asio/detail/array_fwd.hpp>
#include <boost/asio/detail/base_from_completion_cond.hpp>
#include <boost/asio/detail/bind_handler.hpp>
#include <boost/asio/detail/consuming_buffers.hpp>
#include <boost/asio/detail/dependent_type.hpp>
#include <boost/asio/detail/handler_alloc_helpers.hpp>
#include <boost/asio/detail/handler_cont_helpers.hpp>
#include <boost/asio/detail/handler_invoke_helpers.hpp>
#include <boost/asio/detail/handler_tracking.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/throw_error.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

namespace detail
{
template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
typename ConstBufferIterator, typename CompletionCondition>
std::size_t write_at_buffer_sequence(SyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
const ConstBufferIterator&, CompletionCondition completion_condition,
boost::system::error_code& ec)
{
ec = boost::system::error_code();
boost::asio::detail::consuming_buffers<const_buffer,
ConstBufferSequence, ConstBufferIterator> tmp(buffers);
while (!tmp.empty())
{
if (std::size_t max_size = detail::adapt_completion_condition_result(
completion_condition(ec, tmp.total_consumed())))
{
tmp.consume(d.write_some_at(offset + tmp.total_consumed(),
tmp.prepare(max_size), ec));
}
else
break;
}
return tmp.total_consumed();
}
} 

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
typename CompletionCondition>
std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
CompletionCondition completion_condition, boost::system::error_code& ec)
{
return detail::write_at_buffer_sequence(d, offset, buffers,
boost::asio::buffer_sequence_begin(buffers),
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition), ec);
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers)
{
boost::system::error_code ec;
std::size_t bytes_transferred = write_at(
d, offset, buffers, transfer_all(), ec);
boost::asio::detail::throw_error(ec, "write_at");
return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
boost::system::error_code& ec)
{
return write_at(d, offset, buffers, transfer_all(), ec);
}

template <typename SyncRandomAccessWriteDevice, typename ConstBufferSequence,
typename CompletionCondition>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
CompletionCondition completion_condition)
{
boost::system::error_code ec;
std::size_t bytes_transferred = write_at(d, offset, buffers,
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition), ec);
boost::asio::detail::throw_error(ec, "write_at");
return bytes_transferred;
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
#if !defined(BOOST_ASIO_NO_IOSTREAM)

template <typename SyncRandomAccessWriteDevice, typename Allocator,
typename CompletionCondition>
std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, boost::asio::basic_streambuf<Allocator>& b,
CompletionCondition completion_condition, boost::system::error_code& ec)
{
std::size_t bytes_transferred = write_at(d, offset, b.data(),
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition), ec);
b.consume(bytes_transferred);
return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, boost::asio::basic_streambuf<Allocator>& b)
{
boost::system::error_code ec;
std::size_t bytes_transferred = write_at(d, offset, b, transfer_all(), ec);
boost::asio::detail::throw_error(ec, "write_at");
return bytes_transferred;
}

template <typename SyncRandomAccessWriteDevice, typename Allocator>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, boost::asio::basic_streambuf<Allocator>& b,
boost::system::error_code& ec)
{
return write_at(d, offset, b, transfer_all(), ec);
}

template <typename SyncRandomAccessWriteDevice, typename Allocator,
typename CompletionCondition>
inline std::size_t write_at(SyncRandomAccessWriteDevice& d,
uint64_t offset, boost::asio::basic_streambuf<Allocator>& b,
CompletionCondition completion_condition)
{
boost::system::error_code ec;
std::size_t bytes_transferred = write_at(d, offset, b,
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition), ec);
boost::asio::detail::throw_error(ec, "write_at");
return bytes_transferred;
}

#endif 
#endif 

namespace detail
{
template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
class write_at_op
: detail::base_from_completion_cond<CompletionCondition>
{
public:
write_at_op(AsyncRandomAccessWriteDevice& device,
uint64_t offset, const ConstBufferSequence& buffers,
CompletionCondition& completion_condition, WriteHandler& handler)
: detail::base_from_completion_cond<
CompletionCondition>(completion_condition),
device_(device),
offset_(offset),
buffers_(buffers),
start_(0),
handler_(BOOST_ASIO_MOVE_CAST(WriteHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
write_at_op(const write_at_op& other)
: detail::base_from_completion_cond<CompletionCondition>(other),
device_(other.device_),
offset_(other.offset_),
buffers_(other.buffers_),
start_(other.start_),
handler_(other.handler_)
{
}

write_at_op(write_at_op&& other)
: detail::base_from_completion_cond<CompletionCondition>(
BOOST_ASIO_MOVE_CAST(detail::base_from_completion_cond<
CompletionCondition>)(other)),
device_(other.device_),
offset_(other.offset_),
buffers_(BOOST_ASIO_MOVE_CAST(buffers_type)(other.buffers_)),
start_(other.start_),
handler_(BOOST_ASIO_MOVE_CAST(WriteHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
std::size_t bytes_transferred, int start = 0)
{
std::size_t max_size;
switch (start_ = start)
{
case 1:
max_size = this->check_for_completion(ec, buffers_.total_consumed());
do
{
{
BOOST_ASIO_HANDLER_LOCATION((__FILE__, __LINE__, "async_write_at"));
device_.async_write_some_at(
offset_ + buffers_.total_consumed(), buffers_.prepare(max_size),
BOOST_ASIO_MOVE_CAST(write_at_op)(*this));
}
return; default:
buffers_.consume(bytes_transferred);
if ((!ec && bytes_transferred == 0) || buffers_.empty())
break;
max_size = this->check_for_completion(ec, buffers_.total_consumed());
} while (max_size > 0);

handler_(ec, buffers_.total_consumed());
}
}

typedef boost::asio::detail::consuming_buffers<const_buffer,
ConstBufferSequence, ConstBufferIterator> buffers_type;

AsyncRandomAccessWriteDevice& device_;
uint64_t offset_;
buffers_type buffers_;
int start_;
WriteHandler handler_;
};

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
inline bool asio_handler_is_continuation(
write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
{
return this_handler->start_ == 0 ? true
: boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler>
inline void start_write_at_buffer_sequence_op(AsyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
const ConstBufferIterator&, CompletionCondition& completion_condition,
WriteHandler& handler)
{
detail::write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>(
d, offset, buffers, completion_condition, handler)(
boost::system::error_code(), 0, 1);
}

template <typename AsyncRandomAccessWriteDevice>
class initiate_async_write_at_buffer_sequence
{
public:
typedef typename AsyncRandomAccessWriteDevice::executor_type executor_type;

explicit initiate_async_write_at_buffer_sequence(
AsyncRandomAccessWriteDevice& device)
: device_(device)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return device_.get_executor();
}

template <typename WriteHandler, typename ConstBufferSequence,
typename CompletionCondition>
void operator()(BOOST_ASIO_MOVE_ARG(WriteHandler) handler,
uint64_t offset, const ConstBufferSequence& buffers,
BOOST_ASIO_MOVE_ARG(CompletionCondition) completion_cond) const
{
BOOST_ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

non_const_lvalue<WriteHandler> handler2(handler);
non_const_lvalue<CompletionCondition> completion_cond2(completion_cond);
start_write_at_buffer_sequence_op(device_, offset, buffers,
boost::asio::buffer_sequence_begin(buffers),
completion_cond2.value, handler2.value);
}

private:
AsyncRandomAccessWriteDevice& device_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler, typename Allocator>
struct associated_allocator<
detail::write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>,
Allocator>
{
typedef typename associated_allocator<WriteHandler, Allocator>::type type;

static type get(
const detail::write_at_op<AsyncRandomAccessWriteDevice,
ConstBufferSequence, ConstBufferIterator,
CompletionCondition, WriteHandler>& h,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<WriteHandler, Allocator>::get(h.handler_, a);
}
};

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename ConstBufferIterator,
typename CompletionCondition, typename WriteHandler, typename Executor>
struct associated_executor<
detail::write_at_op<AsyncRandomAccessWriteDevice, ConstBufferSequence,
ConstBufferIterator, CompletionCondition, WriteHandler>,
Executor>
: detail::associated_executor_forwarding_base<WriteHandler, Executor>
{
typedef typename associated_executor<WriteHandler, Executor>::type type;

static type get(
const detail::write_at_op<AsyncRandomAccessWriteDevice,
ConstBufferSequence, ConstBufferIterator,
CompletionCondition, WriteHandler>& h,
const Executor& ex = Executor()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<WriteHandler, Executor>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncRandomAccessWriteDevice,
typename ConstBufferSequence, typename CompletionCondition,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
CompletionCondition completion_condition,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler)
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_write_at_buffer_sequence<
AsyncRandomAccessWriteDevice>(d),
handler, offset, buffers,
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition));
}

template <typename AsyncRandomAccessWriteDevice, typename ConstBufferSequence,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
uint64_t offset, const ConstBufferSequence& buffers,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler)
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_write_at_buffer_sequence<
AsyncRandomAccessWriteDevice>(d),
handler, offset, buffers, transfer_all());
}

#if !defined(BOOST_ASIO_NO_EXTENSIONS)
#if !defined(BOOST_ASIO_NO_IOSTREAM)

namespace detail
{
template <typename Allocator, typename WriteHandler>
class write_at_streambuf_op
{
public:
write_at_streambuf_op(
boost::asio::basic_streambuf<Allocator>& streambuf,
WriteHandler& handler)
: streambuf_(streambuf),
handler_(BOOST_ASIO_MOVE_CAST(WriteHandler)(handler))
{
}

#if defined(BOOST_ASIO_HAS_MOVE)
write_at_streambuf_op(const write_at_streambuf_op& other)
: streambuf_(other.streambuf_),
handler_(other.handler_)
{
}

write_at_streambuf_op(write_at_streambuf_op&& other)
: streambuf_(other.streambuf_),
handler_(BOOST_ASIO_MOVE_CAST(WriteHandler)(other.handler_))
{
}
#endif 

void operator()(const boost::system::error_code& ec,
const std::size_t bytes_transferred)
{
streambuf_.consume(bytes_transferred);
handler_(ec, bytes_transferred);
}

boost::asio::basic_streambuf<Allocator>& streambuf_;
WriteHandler handler_;
};

template <typename Allocator, typename WriteHandler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
{
#if defined(BOOST_ASIO_NO_DEPRECATED)
boost_asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return boost_asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename Allocator, typename WriteHandler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
{
boost_asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename Allocator, typename WriteHandler>
inline bool asio_handler_is_continuation(
write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
{
return boost_asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename Allocator, typename WriteHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename Allocator, typename WriteHandler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
write_at_streambuf_op<Allocator, WriteHandler>* this_handler)
{
boost_asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(BOOST_ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename AsyncRandomAccessWriteDevice>
class initiate_async_write_at_streambuf
{
public:
typedef typename AsyncRandomAccessWriteDevice::executor_type executor_type;

explicit initiate_async_write_at_streambuf(
AsyncRandomAccessWriteDevice& device)
: device_(device)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return device_.get_executor();
}

template <typename WriteHandler,
typename Allocator, typename CompletionCondition>
void operator()(BOOST_ASIO_MOVE_ARG(WriteHandler) handler,
uint64_t offset, basic_streambuf<Allocator>* b,
BOOST_ASIO_MOVE_ARG(CompletionCondition) completion_condition) const
{
BOOST_ASIO_WRITE_HANDLER_CHECK(WriteHandler, handler) type_check;

non_const_lvalue<WriteHandler> handler2(handler);
async_write_at(device_, offset, b->data(),
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition),
write_at_streambuf_op<Allocator, typename decay<WriteHandler>::type>(
*b, handler2.value));
}

private:
AsyncRandomAccessWriteDevice& device_;
};
} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename Allocator, typename WriteHandler, typename Allocator1>
struct associated_allocator<
detail::write_at_streambuf_op<Allocator, WriteHandler>,
Allocator1>
{
typedef typename associated_allocator<WriteHandler, Allocator1>::type type;

static type get(
const detail::write_at_streambuf_op<Allocator, WriteHandler>& h,
const Allocator1& a = Allocator1()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<WriteHandler, Allocator1>::get(h.handler_, a);
}
};

template <typename Executor, typename WriteHandler, typename Executor1>
struct associated_executor<
detail::write_at_streambuf_op<Executor, WriteHandler>,
Executor1>
: detail::associated_executor_forwarding_base<WriteHandler, Executor>
{
typedef typename associated_executor<WriteHandler, Executor1>::type type;

static type get(
const detail::write_at_streambuf_op<Executor, WriteHandler>& h,
const Executor1& ex = Executor1()) BOOST_ASIO_NOEXCEPT
{
return associated_executor<WriteHandler, Executor1>::get(h.handler_, ex);
}
};

#endif 

template <typename AsyncRandomAccessWriteDevice,
typename Allocator, typename CompletionCondition,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
uint64_t offset, boost::asio::basic_streambuf<Allocator>& b,
CompletionCondition completion_condition,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler)
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_write_at_streambuf<
AsyncRandomAccessWriteDevice>(d),
handler, offset, &b,
BOOST_ASIO_MOVE_CAST(CompletionCondition)(completion_condition));
}

template <typename AsyncRandomAccessWriteDevice, typename Allocator,
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code,
std::size_t)) WriteHandler>
inline BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (boost::system::error_code, std::size_t))
async_write_at(AsyncRandomAccessWriteDevice& d,
uint64_t offset, boost::asio::basic_streambuf<Allocator>& b,
BOOST_ASIO_MOVE_ARG(WriteHandler) handler)
{
return async_initiate<WriteHandler,
void (boost::system::error_code, std::size_t)>(
detail::initiate_async_write_at_streambuf<
AsyncRandomAccessWriteDevice>(d),
handler, offset, &b, transfer_all());
}

#endif 
#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 