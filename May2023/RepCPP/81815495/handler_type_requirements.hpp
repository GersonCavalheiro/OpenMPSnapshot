
#ifndef ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP
#define ASIO_DETAIL_HANDLER_TYPE_REQUIREMENTS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS)
# if !defined(__GNUC__) || (__GNUC__ >= 4)
#  define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS 1
# endif 
#endif 

#if !defined(ASIO_DISABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)
# if defined(__GNUC__)
#  if ((__GNUC__ == 4) && (__GNUC_MINOR__ >= 5)) || (__GNUC__ > 4)
#   if defined(__GXX_EXPERIMENTAL_CXX0X__)
#    define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#   endif 
#  endif 
# endif 
# if defined(ASIO_MSVC)
#  if (_MSC_VER >= 1600)
#   define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#  endif 
# endif 
# if defined(__clang__)
#  if __has_feature(__cxx_static_assert__)
#   define ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT 1
#  endif 
# endif 
#endif 

#if defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)
# include "asio/async_result.hpp"
#endif 

namespace asio {
namespace detail {

#if defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS)

# if defined(ASIO_ENABLE_HANDLER_TYPE_REQUIREMENTS_ASSERT)

template <typename Handler>
auto zero_arg_copyable_handler_test(Handler h, void*)
-> decltype(
sizeof(Handler(static_cast<const Handler&>(h))),
(ASIO_MOVE_OR_LVALUE(Handler)(h)()),
char(0));

template <typename Handler>
char (&zero_arg_copyable_handler_test(Handler, ...))[2];

template <typename Handler, typename Arg1>
auto one_arg_handler_test(Handler h, Arg1* a1)
-> decltype(
sizeof(Handler(ASIO_MOVE_CAST(Handler)(h))),
(ASIO_MOVE_OR_LVALUE(Handler)(h)(*a1)),
char(0));

template <typename Handler>
char (&one_arg_handler_test(Handler h, ...))[2];

template <typename Handler, typename Arg1, typename Arg2>
auto two_arg_handler_test(Handler h, Arg1* a1, Arg2* a2)
-> decltype(
sizeof(Handler(ASIO_MOVE_CAST(Handler)(h))),
(ASIO_MOVE_OR_LVALUE(Handler)(h)(*a1, *a2)),
char(0));

template <typename Handler>
char (&two_arg_handler_test(Handler, ...))[2];

template <typename Handler, typename Arg1, typename Arg2>
auto two_arg_move_handler_test(Handler h, Arg1* a1, Arg2* a2)
-> decltype(
sizeof(Handler(ASIO_MOVE_CAST(Handler)(h))),
(ASIO_MOVE_OR_LVALUE(Handler)(h)(
*a1, ASIO_MOVE_CAST(Arg2)(*a2))),
char(0));

template <typename Handler>
char (&two_arg_move_handler_test(Handler, ...))[2];

#  define ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT(expr, msg) \
static_assert(expr, msg);

# else 

#  define ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT(expr, msg)

# endif 

template <typename T> T& lvref();
template <typename T> T& lvref(T);
template <typename T> const T& clvref();
template <typename T> const T& clvref(T);
#if defined(ASIO_HAS_MOVE)
template <typename T> T rvref();
template <typename T> T rvref(T);
template <typename T> T rorlvref();
#else 
template <typename T> const T& rvref();
template <typename T> const T& rvref(T);
template <typename T> T& rorlvref();
#endif 
template <typename T> char argbyv(T);

template <int>
struct handler_type_requirements
{
};

#define ASIO_LEGACY_COMPLETION_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void()) asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::zero_arg_copyable_handler_test( \
asio::detail::clvref< \
asio_true_handler_type>(), 0)) == 1, \
"CompletionHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::clvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()(), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_READ_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, std::size_t)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const std::size_t*>(0))) == 1, \
"ReadHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const std::size_t>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_WRITE_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, std::size_t)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const std::size_t*>(0))) == 1, \
"WriteHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const std::size_t>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_ACCEPT_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::one_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0))) == 1, \
"AcceptHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_MOVE_ACCEPT_HANDLER_CHECK( \
handler_type, handler, socket_type) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, socket_type)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_move_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<socket_type*>(0))) == 1, \
"MoveAcceptHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::rvref<socket_type>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_CONNECT_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::one_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0))) == 1, \
"ConnectHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_RANGE_CONNECT_HANDLER_CHECK( \
handler_type, handler, endpoint_type) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, endpoint_type)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const endpoint_type*>(0))) == 1, \
"RangeConnectHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const endpoint_type>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_ITERATOR_CONNECT_HANDLER_CHECK( \
handler_type, handler, iter_type) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, iter_type)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const iter_type*>(0))) == 1, \
"IteratorConnectHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const iter_type>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_RESOLVE_HANDLER_CHECK( \
handler_type, handler, range_type) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, range_type)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const range_type*>(0))) == 1, \
"ResolveHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const range_type>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_WAIT_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::one_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0))) == 1, \
"WaitHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_SIGNAL_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, int)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const int*>(0))) == 1, \
"SignalHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const int>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::one_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0))) == 1, \
"HandshakeHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_BUFFERED_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code, std::size_t)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::two_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0), \
static_cast<const std::size_t*>(0))) == 1, \
"BufferedHandshakeHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>(), \
asio::detail::lvref<const std::size_t>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#define ASIO_SHUTDOWN_HANDLER_CHECK( \
handler_type, handler) \
\
typedef ASIO_HANDLER_TYPE(handler_type, \
void(asio::error_code)) \
asio_true_handler_type; \
\
ASIO_HANDLER_TYPE_REQUIREMENTS_ASSERT( \
sizeof(asio::detail::one_arg_handler_test( \
asio::detail::rvref< \
asio_true_handler_type>(), \
static_cast<const asio::error_code*>(0))) == 1, \
"ShutdownHandler type requirements not met") \
\
typedef asio::detail::handler_type_requirements< \
sizeof( \
asio::detail::argbyv( \
asio::detail::rvref< \
asio_true_handler_type>())) + \
sizeof( \
asio::detail::rorlvref< \
asio_true_handler_type>()( \
asio::detail::lvref<const asio::error_code>()), \
char(0))> ASIO_UNUSED_TYPEDEF

#else 

#define ASIO_LEGACY_COMPLETION_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_READ_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_WRITE_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_ACCEPT_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_MOVE_ACCEPT_HANDLER_CHECK( \
handler_type, handler, socket_type) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_CONNECT_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_RANGE_CONNECT_HANDLER_CHECK( \
handler_type, handler, iter_type) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_ITERATOR_CONNECT_HANDLER_CHECK( \
handler_type, handler, iter_type) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_RESOLVE_HANDLER_CHECK( \
handler_type, handler, iter_type) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_WAIT_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_SIGNAL_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_BUFFERED_HANDSHAKE_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#define ASIO_SHUTDOWN_HANDLER_CHECK( \
handler_type, handler) \
typedef int ASIO_UNUSED_TYPEDEF

#endif 

} 
} 

#endif 
