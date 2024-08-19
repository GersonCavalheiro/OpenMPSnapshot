
#ifndef ASIO_SSL_DETAIL_STREAM_CORE_HPP
#define ASIO_SSL_DETAIL_STREAM_CORE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/deadline_timer.hpp"
#else 
# include "asio/steady_timer.hpp"
#endif 
#include "asio/ssl/detail/engine.hpp"
#include "asio/buffer.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

struct stream_core
{
enum { max_tls_record_size = 17 * 1024 };

template <typename Executor>
stream_core(SSL_CTX* context, const Executor& ex)
: engine_(context),
pending_read_(ex),
pending_write_(ex),
output_buffer_space_(max_tls_record_size),
output_buffer_(asio::buffer(output_buffer_space_)),
input_buffer_space_(max_tls_record_size),
input_buffer_(asio::buffer(input_buffer_space_))
{
pending_read_.expires_at(neg_infin());
pending_write_.expires_at(neg_infin());
}

#if defined(ASIO_HAS_MOVE)
stream_core(stream_core&& other)
: engine_(ASIO_MOVE_CAST(engine)(other.engine_)),
#if defined(ASIO_HAS_BOOST_DATE_TIME)
pending_read_(
ASIO_MOVE_CAST(asio::deadline_timer)(
other.pending_read_)),
pending_write_(
ASIO_MOVE_CAST(asio::deadline_timer)(
other.pending_write_)),
#else 
pending_read_(
ASIO_MOVE_CAST(asio::steady_timer)(
other.pending_read_)),
pending_write_(
ASIO_MOVE_CAST(asio::steady_timer)(
other.pending_write_)),
#endif 
output_buffer_space_(
ASIO_MOVE_CAST(std::vector<unsigned char>)(
other.output_buffer_space_)),
output_buffer_(other.output_buffer_),
input_buffer_space_(
ASIO_MOVE_CAST(std::vector<unsigned char>)(
other.input_buffer_space_)),
input_buffer_(other.input_buffer_),
input_(other.input_)
{
other.output_buffer_ = asio::mutable_buffer(0, 0);
other.input_buffer_ = asio::mutable_buffer(0, 0);
other.input_ = asio::const_buffer(0, 0);
}
#endif 

~stream_core()
{
}

#if defined(ASIO_HAS_MOVE)
stream_core& operator=(stream_core&& other)
{
if (this != &other)
{
engine_ = ASIO_MOVE_CAST(engine)(other.engine_);
#if defined(ASIO_HAS_BOOST_DATE_TIME)
pending_read_ =
ASIO_MOVE_CAST(asio::deadline_timer)(
other.pending_read_);
pending_write_ =
ASIO_MOVE_CAST(asio::deadline_timer)(
other.pending_write_);
#else 
pending_read_ =
ASIO_MOVE_CAST(asio::steady_timer)(
other.pending_read_);
pending_write_ =
ASIO_MOVE_CAST(asio::steady_timer)(
other.pending_write_);
#endif 
output_buffer_space_ =
ASIO_MOVE_CAST(std::vector<unsigned char>)(
other.output_buffer_space_);
output_buffer_ = other.output_buffer_;
input_buffer_space_ =
ASIO_MOVE_CAST(std::vector<unsigned char>)(
other.input_buffer_space_);
input_ = other.input_;
other.output_buffer_ = asio::mutable_buffer(0, 0);
other.input_buffer_ = asio::mutable_buffer(0, 0);
other.input_ = asio::const_buffer(0, 0);
}
return *this;
}
#endif 

engine engine_;

#if defined(ASIO_HAS_BOOST_DATE_TIME)
asio::deadline_timer pending_read_;

asio::deadline_timer pending_write_;

static asio::deadline_timer::time_type neg_infin()
{
return boost::posix_time::neg_infin;
}

static asio::deadline_timer::time_type pos_infin()
{
return boost::posix_time::pos_infin;
}

static asio::deadline_timer::time_type expiry(
const asio::deadline_timer& timer)
{
return timer.expires_at();
}
#else 
asio::steady_timer pending_read_;

asio::steady_timer pending_write_;

static asio::steady_timer::time_point neg_infin()
{
return (asio::steady_timer::time_point::min)();
}

static asio::steady_timer::time_point pos_infin()
{
return (asio::steady_timer::time_point::max)();
}

static asio::steady_timer::time_point expiry(
const asio::steady_timer& timer)
{
return timer.expiry();
}
#endif 

std::vector<unsigned char> output_buffer_space_;

asio::mutable_buffer output_buffer_;

std::vector<unsigned char> input_buffer_space_;

asio::mutable_buffer input_buffer_;

asio::const_buffer input_;
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
