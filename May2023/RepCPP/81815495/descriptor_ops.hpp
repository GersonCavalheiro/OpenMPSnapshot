
#ifndef ASIO_DETAIL_DESCRIPTOR_OPS_HPP
#define ASIO_DETAIL_DESCRIPTOR_OPS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS) \
&& !defined(ASIO_WINDOWS_RUNTIME) \
&& !defined(__CYGWIN__)

#include <cstddef>
#include "asio/error.hpp"
#include "asio/error_code.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {
namespace descriptor_ops {

enum
{
user_set_non_blocking = 1,

internal_non_blocking = 2,

non_blocking = user_set_non_blocking | internal_non_blocking,

possible_dup = 4
};

typedef unsigned char state_type;

inline void get_last_error(
asio::error_code& ec, bool is_error_condition)
{
if (!is_error_condition)
{
ec.assign(0, ec.category());
}
else
{
ec = asio::error_code(errno,
asio::error::get_system_category());
}
}

ASIO_DECL int open(const char* path, int flags,
asio::error_code& ec);

ASIO_DECL int close(int d, state_type& state,
asio::error_code& ec);

ASIO_DECL bool set_user_non_blocking(int d,
state_type& state, bool value, asio::error_code& ec);

ASIO_DECL bool set_internal_non_blocking(int d,
state_type& state, bool value, asio::error_code& ec);

typedef iovec buf;

ASIO_DECL std::size_t sync_read(int d, state_type state, buf* bufs,
std::size_t count, bool all_empty, asio::error_code& ec);

ASIO_DECL std::size_t sync_read1(int d, state_type state, void* data,
std::size_t size, asio::error_code& ec);

ASIO_DECL bool non_blocking_read(int d, buf* bufs, std::size_t count,
asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL bool non_blocking_read1(int d, void* data, std::size_t size,
asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL std::size_t sync_write(int d, state_type state,
const buf* bufs, std::size_t count, bool all_empty,
asio::error_code& ec);

ASIO_DECL std::size_t sync_write1(int d, state_type state,
const void* data, std::size_t size, asio::error_code& ec);

ASIO_DECL bool non_blocking_write(int d,
const buf* bufs, std::size_t count,
asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL bool non_blocking_write1(int d,
const void* data, std::size_t size,
asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL int ioctl(int d, state_type& state, long cmd,
ioctl_arg_type* arg, asio::error_code& ec);

ASIO_DECL int fcntl(int d, int cmd, asio::error_code& ec);

ASIO_DECL int fcntl(int d, int cmd,
long arg, asio::error_code& ec);

ASIO_DECL int poll_read(int d,
state_type state, asio::error_code& ec);

ASIO_DECL int poll_write(int d,
state_type state, asio::error_code& ec);

ASIO_DECL int poll_error(int d,
state_type state, asio::error_code& ec);

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/descriptor_ops.ipp"
#endif 

#endif 

#endif 
