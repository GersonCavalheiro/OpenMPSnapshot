
#ifndef ASIO_TS_NETFWD_HPP
#define ASIO_TS_NETFWD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_CHRONO)
# include "asio/detail/chrono.hpp"
#endif 

#if defined(ASIO_HAS_BOOST_DATE_TIME)
# include "asio/detail/date_time_fwd.hpp"
#endif 

#if !defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)
#include "asio/execution/blocking.hpp"
#include "asio/execution/outstanding_work.hpp"
#include "asio/execution/relationship.hpp"
#endif 

#if !defined(GENERATING_DOCUMENTATION)

#include "asio/detail/push_options.hpp"

namespace asio {

class execution_context;

template <typename T, typename Executor>
class executor_binder;

#if !defined(ASIO_EXECUTOR_WORK_GUARD_DECL)
#define ASIO_EXECUTOR_WORK_GUARD_DECL

template <typename Executor, typename = void, typename = void>
class executor_work_guard;

#endif 

template <typename Blocking, typename Relationship, typename Allocator>
class basic_system_executor;

#if defined(ASIO_USE_TS_EXECUTOR_AS_DEFAULT)

class executor;

typedef executor any_io_executor;

#else 

namespace execution {

#if !defined(ASIO_EXECUTION_ANY_EXECUTOR_FWD_DECL)
#define ASIO_EXECUTION_ANY_EXECUTOR_FWD_DECL

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... SupportableProperties>
class any_executor;

#else 

template <typename = void, typename = void, typename = void,
typename = void, typename = void, typename = void,
typename = void, typename = void, typename = void>
class any_executor;

#endif 

#endif 

template <typename U>
struct context_as_t;

template <typename Property>
struct prefer_only;

} 

class any_io_executor;

#endif 

template <typename Executor>
class strand;

class io_context;

template <typename Clock>
struct wait_traits;

#if defined(ASIO_HAS_BOOST_DATE_TIME)

template <typename Time>
struct time_traits;

#endif 

#if !defined(ASIO_BASIC_WAITABLE_TIMER_FWD_DECL)
#define ASIO_BASIC_WAITABLE_TIMER_FWD_DECL

template <typename Clock,
typename WaitTraits = wait_traits<Clock>,
typename Executor = any_io_executor>
class basic_waitable_timer;

#endif 

#if defined(ASIO_HAS_CHRONO)

typedef basic_waitable_timer<chrono::system_clock> system_timer;

typedef basic_waitable_timer<chrono::steady_clock> steady_timer;

typedef basic_waitable_timer<chrono::high_resolution_clock>
high_resolution_timer;

#endif 

#if !defined(ASIO_BASIC_SOCKET_FWD_DECL)
#define ASIO_BASIC_SOCKET_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_socket;

#endif 

#if !defined(ASIO_BASIC_DATAGRAM_SOCKET_FWD_DECL)
#define ASIO_BASIC_DATAGRAM_SOCKET_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_datagram_socket;

#endif 

#if !defined(ASIO_BASIC_STREAM_SOCKET_FWD_DECL)
#define ASIO_BASIC_STREAM_SOCKET_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_stream_socket;

#endif 

#if !defined(ASIO_BASIC_SOCKET_ACCEPTOR_FWD_DECL)
#define ASIO_BASIC_SOCKET_ACCEPTOR_FWD_DECL

template <typename Protocol, typename Executor = any_io_executor>
class basic_socket_acceptor;

#endif 

#if !defined(ASIO_BASIC_SOCKET_STREAMBUF_FWD_DECL)
#define ASIO_BASIC_SOCKET_STREAMBUF_FWD_DECL

template <typename Protocol,
#if defined(ASIO_HAS_BOOST_DATE_TIME) \
|| defined(GENERATING_DOCUMENTATION)
typename Clock = boost::posix_time::ptime,
typename WaitTraits = time_traits<Clock> >
#else
typename Clock = chrono::steady_clock,
typename WaitTraits = wait_traits<Clock> >
#endif
class basic_socket_streambuf;

#endif 

#if !defined(ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL)
#define ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL

template <typename Protocol,
#if defined(ASIO_HAS_BOOST_DATE_TIME) \
|| defined(GENERATING_DOCUMENTATION)
typename Clock = boost::posix_time::ptime,
typename WaitTraits = time_traits<Clock> >
#else
typename Clock = chrono::steady_clock,
typename WaitTraits = wait_traits<Clock> >
#endif
class basic_socket_iostream;

#endif 

namespace ip {

class address;

class address_v4;

class address_v6;

template <typename Address>
class basic_address_iterator;

typedef basic_address_iterator<address_v4> address_v4_iterator;

typedef basic_address_iterator<address_v6> address_v6_iterator;

template <typename Address>
class basic_address_range;

typedef basic_address_range<address_v4> address_v4_range;

typedef basic_address_range<address_v6> address_v6_range;

class network_v4;

class network_v6;

template <typename InternetProtocol>
class basic_endpoint;

template <typename InternetProtocol>
class basic_resolver_entry;

template <typename InternetProtocol>
class basic_resolver_results;

#if !defined(ASIO_IP_BASIC_RESOLVER_FWD_DECL)
#define ASIO_IP_BASIC_RESOLVER_FWD_DECL

template <typename InternetProtocol, typename Executor = any_io_executor>
class basic_resolver;

#endif 

class tcp;

class udp;

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
