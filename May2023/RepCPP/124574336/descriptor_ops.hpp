
#ifndef BOOST_ASIO_DETAIL_DESCRIPTOR_OPS_HPP
#define BOOST_ASIO_DETAIL_DESCRIPTOR_OPS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_WINDOWS) \
&& !defined(BOOST_ASIO_WINDOWS_RUNTIME) \
&& !defined(__CYGWIN__)

#include <cstddef>
#include <boost/asio/error.hpp>
#include <boost/system/error_code.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
boost::system::error_code& ec, bool is_error_condition)
{
if (!is_error_condition)
{
ec.assign(0, ec.category());
}
else
{
ec = boost::system::error_code(errno,
boost::asio::error::get_system_category());
}
}

BOOST_ASIO_DECL int open(const char* path, int flags,
boost::system::error_code& ec);

BOOST_ASIO_DECL int close(int d, state_type& state,
boost::system::error_code& ec);

BOOST_ASIO_DECL bool set_user_non_blocking(int d,
state_type& state, bool value, boost::system::error_code& ec);

BOOST_ASIO_DECL bool set_internal_non_blocking(int d,
state_type& state, bool value, boost::system::error_code& ec);

typedef iovec buf;

BOOST_ASIO_DECL std::size_t sync_read(int d, state_type state, buf* bufs,
std::size_t count, bool all_empty, boost::system::error_code& ec);

BOOST_ASIO_DECL std::size_t sync_read1(int d, state_type state, void* data,
std::size_t size, boost::system::error_code& ec);

BOOST_ASIO_DECL bool non_blocking_read(int d, buf* bufs, std::size_t count,
boost::system::error_code& ec, std::size_t& bytes_transferred);

BOOST_ASIO_DECL bool non_blocking_read1(int d, void* data, std::size_t size,
boost::system::error_code& ec, std::size_t& bytes_transferred);

BOOST_ASIO_DECL std::size_t sync_write(int d, state_type state,
const buf* bufs, std::size_t count, bool all_empty,
boost::system::error_code& ec);

BOOST_ASIO_DECL std::size_t sync_write1(int d, state_type state,
const void* data, std::size_t size, boost::system::error_code& ec);

BOOST_ASIO_DECL bool non_blocking_write(int d,
const buf* bufs, std::size_t count,
boost::system::error_code& ec, std::size_t& bytes_transferred);

BOOST_ASIO_DECL bool non_blocking_write1(int d,
const void* data, std::size_t size,
boost::system::error_code& ec, std::size_t& bytes_transferred);

BOOST_ASIO_DECL int ioctl(int d, state_type& state, long cmd,
ioctl_arg_type* arg, boost::system::error_code& ec);

BOOST_ASIO_DECL int fcntl(int d, int cmd, boost::system::error_code& ec);

BOOST_ASIO_DECL int fcntl(int d, int cmd,
long arg, boost::system::error_code& ec);

BOOST_ASIO_DECL int poll_read(int d,
state_type state, boost::system::error_code& ec);

BOOST_ASIO_DECL int poll_write(int d,
state_type state, boost::system::error_code& ec);

BOOST_ASIO_DECL int poll_error(int d,
state_type state, boost::system::error_code& ec);

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/descriptor_ops.ipp>
#endif 

#endif 

#endif 
