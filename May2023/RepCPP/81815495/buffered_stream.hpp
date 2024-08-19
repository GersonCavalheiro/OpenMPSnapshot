
#ifndef ASIO_BUFFERED_STREAM_HPP
#define ASIO_BUFFERED_STREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>
#include "asio/async_result.hpp"
#include "asio/buffered_read_stream.hpp"
#include "asio/buffered_write_stream.hpp"
#include "asio/buffered_stream_fwd.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {


template <typename Stream>
class buffered_stream
: private noncopyable
{
public:
typedef typename remove_reference<Stream>::type next_layer_type;

typedef typename next_layer_type::lowest_layer_type lowest_layer_type;

typedef typename lowest_layer_type::executor_type executor_type;

template <typename Arg>
explicit buffered_stream(Arg& a)
: inner_stream_impl_(a),
stream_impl_(inner_stream_impl_)
{
}

template <typename Arg>
explicit buffered_stream(Arg& a, std::size_t read_buffer_size,
std::size_t write_buffer_size)
: inner_stream_impl_(a, write_buffer_size),
stream_impl_(inner_stream_impl_, read_buffer_size)
{
}

next_layer_type& next_layer()
{
return stream_impl_.next_layer().next_layer();
}

lowest_layer_type& lowest_layer()
{
return stream_impl_.lowest_layer();
}

const lowest_layer_type& lowest_layer() const
{
return stream_impl_.lowest_layer();
}

executor_type get_executor() ASIO_NOEXCEPT
{
return stream_impl_.lowest_layer().get_executor();
}

void close()
{
stream_impl_.close();
}

ASIO_SYNC_OP_VOID close(asio::error_code& ec)
{
stream_impl_.close(ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}

std::size_t flush()
{
return stream_impl_.next_layer().flush();
}

std::size_t flush(asio::error_code& ec)
{
return stream_impl_.next_layer().flush(ec);
}

template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) WriteHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WriteHandler,
void (asio::error_code, std::size_t))
async_flush(
ASIO_MOVE_ARG(WriteHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return stream_impl_.next_layer().async_flush(
ASIO_MOVE_CAST(WriteHandler)(handler));
}

template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers)
{
return stream_impl_.write_some(buffers);
}

template <typename ConstBufferSequence>
std::size_t write_some(const ConstBufferSequence& buffers,
asio::error_code& ec)
{
return stream_impl_.write_some(buffers, ec);
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
return stream_impl_.async_write_some(buffers,
ASIO_MOVE_CAST(WriteHandler)(handler));
}

std::size_t fill()
{
return stream_impl_.fill();
}

std::size_t fill(asio::error_code& ec)
{
return stream_impl_.fill(ec);
}

template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code,
std::size_t)) ReadHandler
ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(ReadHandler,
void (asio::error_code, std::size_t))
async_fill(
ASIO_MOVE_ARG(ReadHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return stream_impl_.async_fill(ASIO_MOVE_CAST(ReadHandler)(handler));
}

template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers)
{
return stream_impl_.read_some(buffers);
}

template <typename MutableBufferSequence>
std::size_t read_some(const MutableBufferSequence& buffers,
asio::error_code& ec)
{
return stream_impl_.read_some(buffers, ec);
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
return stream_impl_.async_read_some(buffers,
ASIO_MOVE_CAST(ReadHandler)(handler));
}

template <typename MutableBufferSequence>
std::size_t peek(const MutableBufferSequence& buffers)
{
return stream_impl_.peek(buffers);
}

template <typename MutableBufferSequence>
std::size_t peek(const MutableBufferSequence& buffers,
asio::error_code& ec)
{
return stream_impl_.peek(buffers, ec);
}

std::size_t in_avail()
{
return stream_impl_.in_avail();
}

std::size_t in_avail(asio::error_code& ec)
{
return stream_impl_.in_avail(ec);
}

private:
typedef buffered_write_stream<Stream> write_stream_type;
write_stream_type inner_stream_impl_;

typedef buffered_read_stream<write_stream_type&> read_stream_type;
read_stream_type stream_impl_;
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
