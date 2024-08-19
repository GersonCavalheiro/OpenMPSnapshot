
#ifndef BOOST_ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP
#define BOOST_ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS)
# if !defined(__GNUC__) || (__GNUC__ >= 4)
#  define BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS 1
# endif 
#endif 

#if !defined(BOOST_ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)
# if defined(__GNUC__)
#  if ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5)) || (__GNUC__ > 4)
#   if defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#   endif 
#  endif 
# endif 
# if defined(BOOST_ASIO_MSVC)
#  if (_MSC_VER >= 1600)
#   define BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#  endif 
# endif 
# if defined(__clang__)
#  if __has_feature(__cxx_static_assert__)
#   define BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#  endif 
# endif 
#endif 

#if defined(BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)
# include <boost/asio/async_result.hpp>
#endif 

namespace boost {
namespace asio {
namespace detail {

#if defined(BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)

# if defined(BOOST_ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)

template <typename Handler>
auto zero_arg_copyable_handler_test(Handler h, void*)
-> decltype(
sizeof(Handler(static_cast<const Handler&>(h))),
((h)()),
char(0));

template <typename Handler>
char (&zero_arg_copyable_handler_test(Handler, ...))[2];

template <typename Handler, typename Arg1>
auto one_arg_handler_test(Handler h, Arg1* a1)
-> decltype(
sizeof(Handler(BOOST_ASIO_MOVE_CAST(Handler)(h))),
((h)(*a1)),
char(0));

template <typename Handler>
char (&one_arg_handler_test(Handler h, ...))[2];

template <typename Handler, typename Arg1, typename Arg2>
auto two_arg_handler_test(Handler h, Arg1* a1, Arg2* a2)
-> decltype(
sizeof(Handler(BOOST_ASIO_MOVE_CAST(Handler)(h))),
((h)(*a1, *a2)),
char(0));

template <typename Handler>
char (&two_arg_handler_test(Handler, ...))[2];

template <typename Handler, typename Arg1, typename Arg2>
auto two_arg_move_handler_test(Handler h, Arg1* a1, Arg2* a2)
-> decltype(
sizeof(Handler(BOOST_ASIO_MOVE_CAST(Handler)(h))),
((h)(*a1, BOOST_ASIO_MOVE_CAST(Arg2)(*a2))),
char(0));

template <typename Handler>
char (&two_arg_move_handler_test(Handler, ...))[2];

#  define BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT(expr, msg) \
static_assert(expr, msg);

# else 

#  define BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT(expr, msg)

# endif 

template <typename T> T& lvref();
template <typename T> T& lvref(T);
template <typename T> const T& clvref();
template <typename T> const T& clvref(T);
#if defined(BOOST_ASIO_HAS_MOVE)
template <typename T> T rvref();
template <typename T> T rvref(T);
#else 
template <typename T> const T& rvref();
template <typename T> const T& rvref(T);
#endif 
template <typename T> char argbyv(T);

template <int>
struct handler_type_requirements
{
};

#define BOOST_ASIO_LEGACY_COMPLETION_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void()) asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::zero_arg_copyable_handler_test( \
boost::asio::detail::clvref< \
asio_true_handler_type>(), 0)) == 1, \
"CompletionHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::clvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()(), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_READ_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, std::size_t)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const std::size_t*>(0))) == 1, \
"ReadHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const std::size_t>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_WRITE_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, std::size_t)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const std::size_t*>(0))) == 1, \
"WriteHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const std::size_t>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_ACCEPT_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::one_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0))) == 1, \
"AcceptHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_MOVE_ACCEPT_HANDLER_CHECK( \
handler_type, handler, socket_type) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, socket_type)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_move_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<socket_type*>(0))) == 1, \
"MoveAcceptHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::rvref<socket_type>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_CONNECT_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::one_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0))) == 1, \
"ConnectHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_RANGE_CONNECT_HANDLER_CHECK( \
handler_type, handler, endpoint_type) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, endpoint_type)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const endpoint_type*>(0))) == 1, \
"RangeConnectHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const endpoint_type>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_ITERATOR_CONNECT_HANDLER_CHECK( \
handler_type, handler, iter_type) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, iter_type)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const iter_type*>(0))) == 1, \
"IteratorConnectHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const iter_type>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_RESOLVE_HANDLER_CHECK( \
handler_type, handler, range_type) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, range_type)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const range_type*>(0))) == 1, \
"ResolveHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const range_type>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_WAIT_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::one_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0))) == 1, \
"WaitHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_SIGNAL_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, int)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const int*>(0))) == 1, \
"SignalHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const int>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::one_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0))) == 1, \
"HandshakeHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_BUFFERED_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code, std::size_t)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::two_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0), \
static_cast<const std::size_t*>(0))) == 1, \
"BufferedHandshakeHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>(), \
boost::asio::detail::lvref<const std::size_t>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_SHUTDOWN_HANDLER_CHECK( \
handler_type, handler) \
\
typedef BOOST_ASIO_HANDLER_TYPE(handler_type, \
void(boost::system::error_code)) \
asio_true_handler_type; \
\
BOOST_ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(boost::asio::detail::one_arg_handler_test( \
boost::asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const boost::system::error_code*>(0))) == 1, \
"ShutdownHandler type requirements not met") \
\
typedef boost::asio::detail::handler_type_requirements< \
sizeof( \
boost::asio::detail::argbyv( \
boost::asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
boost::asio::detail::lvref< \
asio_true_handler_type>()( \
boost::asio::detail::lvref<const boost::system::error_code>()), \
char(0))> BOOST_ASIO_UNUSED_TYPEDEF

#else 

#define BOOST_ASIO_LEGACY_COMPLETION_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_READ_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_WRITE_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_ACCEPT_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_MOVE_ACCEPT_HANDLER_CHECK( \
handler_type, handler, socket_type) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_CONNECT_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_RANGE_CONNECT_HANDLER_CHECK( \
handler_type, handler, iter_type) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_ITERATOR_CONNECT_HANDLER_CHECK( \
handler_type, handler, iter_type) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_RESOLVE_HANDLER_CHECK( \
handler_type, handler, iter_type) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_WAIT_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_SIGNAL_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_BUFFERED_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#define BOOST_ASIO_SHUTDOWN_HANDLER_CHECK( \
handler_type, handler) \
typedef int BOOST_ASIO_UNUSED_TYPEDEF

#endif 

} 
} 
} 

#endif 
