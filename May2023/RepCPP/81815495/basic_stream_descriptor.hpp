
#ifndef ASIO_POSIX_BASIC_STREAM_DESCRIPTOR_HPP
#define ASIO_POSIX_BASIC_STREAM_DESCRIPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/posix/basic_descriptor.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace posix {


template <typename Executor = any_io_executor>
class basic_stream_descriptor
: public basic_descriptor<Executor>
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_stream_descriptor<Executor1> other;
};

typedef typename basic_descriptor<Executor>::native_handle_type
native_handle_type;


explicit basic_stream_descriptor(const executor_type& ex)
: basic_descriptor<Executor>(ex)
{
}


template <typename ExecutionContext>
explicit basic_stream_descriptor(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: basic_descriptor<Executor>(context)
{
}


basic_stream_descriptor(const executor_type& ex,
const native_handle_type& native_descriptor)
: basic_descriptor<Executor>(ex, native_descriptor)
{
}


template <typename ExecutionContext>
basic_stream_descriptor(ExecutionContext& context,
const native_handle_type& native_descriptor,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: basic_descriptor<Executor>(context, native_descriptor)
{
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_stream_descriptor(basic_stream_descriptor&& other) ASIO_NOEXCEPT
: basic_descriptor<Executor>(std::move(other))
{
}


basic_stream_descriptor& operator=(basic_stream_descriptor&& other)
{
basic_descriptor<Executor>::operator=(std::move(other));
return *this;
}
#endif 


template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers)
{
asio::error_code ec;
std::size_t s = this->impl_.get_service().write_some(
this->impl_.get_implementation(), buffers, ec);
asio::detail::throw_error(ec, "write_some");
return s;
}


template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers,
asio::error_code& ec)
{
return this->impl_.get_service().write_some(
this->impl_.get_implementation(), buffers, ec);
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
std::size_t s = this->impl_.get_service().read_some(
this->impl_.get_implementation(), buffers, ec);
asio::detail::throw_error(ec, "read_some");
return s;
}


template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers,
asio::error_code& ec)
{
return this->impl_.get_service().read_some(
this->impl_.get_implementation(), buffers, ec);
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
class initiate_async_write_some
{
public:
typedef Executor executor_type;

explicit initiate_async_write_some(basic_stream_descriptor* self)
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
basic_stream_descriptor* self_;
};

class initiate_async_read_some
{
public:
typedef Executor executor_type;

explicit initiate_async_read_some(basic_stream_descriptor* self)
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
basic_stream_descriptor* self_;
};
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
