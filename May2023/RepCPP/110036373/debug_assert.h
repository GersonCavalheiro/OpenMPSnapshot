
#ifndef DEBUG_ASSERT_HPP_INCLUDED
#define DEBUG_ASSERT_HPP_INCLUDED

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wassume"
#endif 

#include <cstdlib>

#ifndef DEBUG_ASSERT_NO_STDIO
#include <cstdio>
#endif

#ifndef DEBUG_ASSERT_MARK_UNREACHABLE
#ifdef __GNUC__
#define DEBUG_ASSERT_MARK_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
#define DEBUG_ASSERT_MARK_UNREACHABLE __assume(0)
#else
#define DEBUG_ASSERT_MARK_UNREACHABLE
#endif
#endif

#ifndef DEBUG_ASSERT_PURE_FUNCTION
#ifdef __GNUC__
#define DEBUG_ASSERT_PURE_FUNCTION __attribute__((pure))
#else
#define DEBUG_ASSERT_PURE_FUNCTION
#endif
#endif

#if !defined(DEBUG_ASSERT_ASSUME) && defined(__clang__)
#if __has_builtin(__builtin_assume)
#define DEBUG_ASSERT_ASSUME(Expr) static_cast<void>(__builtin_assume(Expr), 0)
#endif
#endif

#ifndef DEBUG_ASSERT_ASSUME
#ifdef __GNUC__
#define DEBUG_ASSERT_ASSUME(Expr) static_cast<void>((Expr) ? 0 : (__builtin_unreachable(), 0))
#elif defined(_MSC_VER)
#define DEBUG_ASSERT_ASSUME(Expr) static_cast<void>(__assume(Expr), 0)
#else
#define DEBUG_ASSERT_ASSUME(Expr) static_cast<void>(0)
#endif
#endif

#ifndef DEBUG_ASSERT_FORCE_INLINE
#ifdef __GNUC__
#define DEBUG_ASSERT_FORCE_INLINE [[gnu::always_inline]] inline
#elif defined(_MSC_VER)
#define DEBUG_ASSERT_FORCE_INLINE __forceinline
#else
#define DEBUG_ASSERT_FORCE_INLINE inline
#endif
#endif

namespace debug_assert
{
struct source_location
{
const char* file_name;   
unsigned    line_number; 
};

#define DEBUG_ASSERT_CUR_SOURCE_LOCATION                                                           \
debug_assert::source_location                                                                  \
{                                                                                              \
__FILE__, static_cast<unsigned>(__LINE__)                                                  \
}

template <unsigned Level>
struct level
{
};

template <unsigned Level>
struct set_level
{
static const unsigned level = Level;
};

template <unsigned Level>
const unsigned set_level<Level>::level;

struct allow_exception
{
static const bool throwing_exception_is_allowed = true;
};

struct no_handler
{
template <typename... Args>
static void handle(const source_location&, const char*, Args&&...) noexcept
{
}
};

struct default_handler
{
static void handle(const source_location& loc, const char* expression,
const char* message = nullptr) noexcept
{
#ifndef DEBUG_ASSERT_NO_STDIO
if (*expression == '\0')
{
if (message)
std::fprintf(stderr, "[debug assert] %s:%u: Unreachable code reached - %s.\n",
loc.file_name, loc.line_number, message);
else
std::fprintf(stderr, "[debug assert] %s:%u: Unreachable code reached.\n",
loc.file_name, loc.line_number);
}
else if (message)
std::fprintf(stderr, "[debug assert] %s:%u: Assertion '%s' failed - %s.\n",
loc.file_name, loc.line_number, expression, message);
else
std::fprintf(stderr, "[debug assert] %s:%u: Assertion '%s' failed.\n",
loc.file_name, loc.line_number, expression);
#else
(void)loc;
(void)expression;
(void)message;
#endif
}
};

namespace detail
{
template <typename T>
struct remove_reference
{
using type = T;
};

template <typename T>
struct remove_reference<T&>
{
using type = T;
};

template <typename T>
struct remove_reference<T&&>
{
using type = T;
};

template <class T>
T&& forward(typename remove_reference<T>::type& t)
{
return static_cast<T&&>(t);
}

template <class T>
T&& forward(typename remove_reference<T>::type&& t)
{
return static_cast<T&&>(t);
}

template <bool Value, typename T = void>
struct enable_if;

template <typename T>
struct enable_if<true, T>
{
using type = T;
};

template <typename T>
struct enable_if<false, T>
{
};

template <class Handler, typename = void>
struct allows_exception
{
static const bool value = false;
};

template <class Handler>
struct allows_exception<Handler,
typename enable_if<Handler::throwing_exception_is_allowed>::type>
{
static const bool value = Handler::throwing_exception_is_allowed;
};

struct regular_void
{
constexpr regular_void() = default;

template <typename T>
constexpr operator T&() const noexcept
{
return DEBUG_ASSERT_MARK_UNREACHABLE, *static_cast<T*>(nullptr);
}
};

template <class Handler, typename... Args>
regular_void debug_assertion_failed(const source_location& loc, const char* expression,
Args&&... args)
{
return Handler::handle(loc, expression, detail::forward<Args>(args)...), std::abort(),
regular_void();
}

template <class Expr, class Handler, unsigned Level, typename... Args>
constexpr auto do_assert(
const Expr& expr, const source_location& loc, const char* expression, Handler,
level<Level>,
Args&&... args) noexcept(!allows_exception<Handler>::value
|| noexcept(Handler::handle(loc, expression,
detail::forward<Args>(args)...)))
-> typename enable_if<Level <= Handler::level, regular_void>::type
{
static_assert(Level > 0, "level of an assertion must not be 0");
return expr() ? regular_void() :
debug_assertion_failed<Handler>(loc, expression,
detail::forward<Args>(args)...);
}

template <class Expr, class Handler, unsigned Level, typename... Args>
DEBUG_ASSERT_FORCE_INLINE constexpr auto do_assert(const Expr& expr, const source_location&,
const char*, Handler, level<Level>,
Args&&...) noexcept ->
typename enable_if<(Level > Handler::level), regular_void>::type
{
return DEBUG_ASSERT_ASSUME(expr()), regular_void();
}

template <class Expr, class Handler, typename... Args>
constexpr auto do_assert(
const Expr& expr, const source_location& loc, const char* expression, Handler,
Args&&... args) noexcept(!allows_exception<Handler>::value
|| noexcept(Handler::handle(loc, expression,
detail::forward<Args>(args)...)))
-> typename enable_if<Handler::level != 0, regular_void>::type
{
return expr() ? regular_void() :
debug_assertion_failed<Handler>(loc, expression,
detail::forward<Args>(args)...);
}

template <class Expr, class Handler, typename... Args>
DEBUG_ASSERT_FORCE_INLINE constexpr auto do_assert(const Expr& expr, const source_location&,
const char*, Handler, Args&&...) noexcept
-> typename enable_if<Handler::level == 0, regular_void>::type
{
return DEBUG_ASSERT_ASSUME(expr()), regular_void();
}

DEBUG_ASSERT_PURE_FUNCTION constexpr bool always_false() noexcept
{
return false;
}
} 
} 

#ifndef DEBUG_ASSERT_DISABLE
#define DEBUG_ASSERT(Expr, ...)                                                                    \
static_cast<void>(                                                                             \
debug_assert::detail::do_assert([&]()                                                      \
DEBUG_ASSERT_PURE_FUNCTION noexcept { return Expr; },  \
DEBUG_ASSERT_CUR_SOURCE_LOCATION, #Expr, __VA_ARGS__))

#define DEBUG_UNREACHABLE(...)                                                                     \
debug_assert::detail::do_assert(debug_assert::detail::always_false,                            \
DEBUG_ASSERT_CUR_SOURCE_LOCATION, "", __VA_ARGS__)
#else
#define DEBUG_ASSERT(Expr, ...) DEBUG_ASSERT_ASSUME(Expr)

#define DEBUG_UNREACHABLE(...) (DEBUG_ASSERT_MARK_UNREACHABLE, debug_assert::detail::regular_void())
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif 

#endif 
