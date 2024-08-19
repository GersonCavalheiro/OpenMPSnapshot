
#ifndef ASIO_SOCKET_BASE_HPP
#define ASIO_SOCKET_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/io_control.hpp"
#include "asio/detail/socket_option.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

class socket_base
{
public:
enum shutdown_type
{
#if defined(GENERATING_DOCUMENTATION)
shutdown_receive = implementation_defined,

shutdown_send = implementation_defined,

shutdown_both = implementation_defined
#else
shutdown_receive = ASIO_OS_DEF(SHUT_RD),
shutdown_send = ASIO_OS_DEF(SHUT_WR),
shutdown_both = ASIO_OS_DEF(SHUT_RDWR)
#endif
};

typedef int message_flags;

#if defined(GENERATING_DOCUMENTATION)
static const int message_peek = implementation_defined;

static const int message_out_of_band = implementation_defined;

static const int message_do_not_route = implementation_defined;

static const int message_end_of_record = implementation_defined;
#else
ASIO_STATIC_CONSTANT(int,
message_peek = ASIO_OS_DEF(MSG_PEEK));
ASIO_STATIC_CONSTANT(int,
message_out_of_band = ASIO_OS_DEF(MSG_OOB));
ASIO_STATIC_CONSTANT(int,
message_do_not_route = ASIO_OS_DEF(MSG_DONTROUTE));
ASIO_STATIC_CONSTANT(int,
message_end_of_record = ASIO_OS_DEF(MSG_EOR));
#endif


enum wait_type
{
wait_read,

wait_write,

wait_error
};


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined broadcast;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_BROADCAST)>
broadcast;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined debug;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_DEBUG)> debug;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined do_not_route;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_DONTROUTE)>
do_not_route;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined keep_alive;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_KEEPALIVE)> keep_alive;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined send_buffer_size;
#else
typedef asio::detail::socket_option::integer<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_SNDBUF)>
send_buffer_size;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined send_low_watermark;
#else
typedef asio::detail::socket_option::integer<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_SNDLOWAT)>
send_low_watermark;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined receive_buffer_size;
#else
typedef asio::detail::socket_option::integer<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_RCVBUF)>
receive_buffer_size;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined receive_low_watermark;
#else
typedef asio::detail::socket_option::integer<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_RCVLOWAT)>
receive_low_watermark;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined reuse_address;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_REUSEADDR)>
reuse_address;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined linger;
#else
typedef asio::detail::socket_option::linger<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_LINGER)>
linger;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined out_of_band_inline;
#else
typedef asio::detail::socket_option::boolean<
ASIO_OS_DEF(SOL_SOCKET), ASIO_OS_DEF(SO_OOBINLINE)>
out_of_band_inline;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined enable_connection_aborted;
#else
typedef asio::detail::socket_option::boolean<
asio::detail::custom_socket_option_level,
asio::detail::enable_connection_aborted_option>
enable_connection_aborted;
#endif


#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined bytes_readable;
#else
typedef asio::detail::io_control::bytes_readable bytes_readable;
#endif

#if defined(GENERATING_DOCUMENTATION)
static const int max_listen_connections = implementation_defined;
#else
ASIO_STATIC_CONSTANT(int, max_listen_connections
= ASIO_OS_DEF(SOMAXCONN));
#endif

#if !defined(ASIO_NO_DEPRECATED)
#if defined(GENERATING_DOCUMENTATION)
static const int max_connections = implementation_defined;
#else
ASIO_STATIC_CONSTANT(int, max_connections
= ASIO_OS_DEF(SOMAXCONN));
#endif
#endif 

protected:
~socket_base()
{
}
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
