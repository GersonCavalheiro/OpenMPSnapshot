

#ifndef ASIO_IMPL_REDIRECT_ERROR_HPP
#define ASIO_IMPL_REDIRECT_ERROR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/associator.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_alloc_helpers.hpp"
#include "asio/detail/handler_cont_helpers.hpp"
#include "asio/detail/handler_invoke_helpers.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/system_error.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Handler>
class redirect_error_handler
{
public:
typedef void result_type;

template <typename CompletionToken>
redirect_error_handler(redirect_error_t<CompletionToken> e)
: ec_(e.ec_),
handler_(ASIO_MOVE_CAST(CompletionToken)(e.token_))
{
}

template <typename RedirectedHandler>
redirect_error_handler(asio::error_code& ec,
ASIO_MOVE_ARG(RedirectedHandler) h)
: ec_(ec),
handler_(ASIO_MOVE_CAST(RedirectedHandler)(h))
{
}

void operator()()
{
ASIO_MOVE_OR_LVALUE(Handler)(handler_)();
}

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Arg, typename... Args>
typename enable_if<
!is_same<typename decay<Arg>::type, asio::error_code>::value
>::type
operator()(ASIO_MOVE_ARG(Arg) arg, ASIO_MOVE_ARG(Args)... args)
{
ASIO_MOVE_OR_LVALUE(Handler)(handler_)(
ASIO_MOVE_CAST(Arg)(arg),
ASIO_MOVE_CAST(Args)(args)...);
}

template <typename... Args>
void operator()(const asio::error_code& ec,
ASIO_MOVE_ARG(Args)... args)
{
ec_ = ec;
ASIO_MOVE_OR_LVALUE(Handler)(handler_)(
ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Arg>
typename enable_if<
!is_same<typename decay<Arg>::type, asio::error_code>::value
>::type
operator()(ASIO_MOVE_ARG(Arg) arg)
{
ASIO_MOVE_OR_LVALUE(Handler)(handler_)(
ASIO_MOVE_CAST(Arg)(arg));
}

void operator()(const asio::error_code& ec)
{
ec_ = ec;
ASIO_MOVE_OR_LVALUE(Handler)(handler_)();
}

#define ASIO_PRIVATE_REDIRECT_ERROR_DEF(n) \
template <typename Arg, ASIO_VARIADIC_TPARAMS(n)> \
typename enable_if< \
!is_same<typename decay<Arg>::type, asio::error_code>::value \
>::type \
operator()(ASIO_MOVE_ARG(Arg) arg, ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
ASIO_MOVE_OR_LVALUE(Handler)(handler_)( \
ASIO_MOVE_CAST(Arg)(arg), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <ASIO_VARIADIC_TPARAMS(n)> \
void operator()(const asio::error_code& ec, \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
ec_ = ec; \
ASIO_MOVE_OR_LVALUE(Handler)(handler_)( \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_REDIRECT_ERROR_DEF)
#undef ASIO_PRIVATE_REDIRECT_ERROR_DEF

#endif 

asio::error_code& ec_;
Handler handler_;
};

template <typename Handler>
inline asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size,
redirect_error_handler<Handler>* this_handler)
{
#if defined(ASIO_NO_DEPRECATED)
asio_handler_alloc_helpers::allocate(size, this_handler->handler_);
return asio_handler_allocate_is_no_longer_used();
#else 
return asio_handler_alloc_helpers::allocate(
size, this_handler->handler_);
#endif 
}

template <typename Handler>
inline asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size,
redirect_error_handler<Handler>* this_handler)
{
asio_handler_alloc_helpers::deallocate(
pointer, size, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_deallocate_is_no_longer_used();
#endif 
}

template <typename Handler>
inline bool asio_handler_is_continuation(
redirect_error_handler<Handler>* this_handler)
{
return asio_handler_cont_helpers::is_continuation(
this_handler->handler_);
}

template <typename Function, typename Handler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function,
redirect_error_handler<Handler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function, typename Handler>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function,
redirect_error_handler<Handler>* this_handler)
{
asio_handler_invoke_helpers::invoke(
function, this_handler->handler_);
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Signature>
struct redirect_error_signature
{
typedef Signature type;
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename R, typename... Args>
struct redirect_error_signature<R(asio::error_code, Args...)>
{
typedef R type(Args...);
};

template <typename R, typename... Args>
struct redirect_error_signature<R(const asio::error_code&, Args...)>
{
typedef R type(Args...);
};

# if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)

template <typename R, typename... Args>
struct redirect_error_signature<R(asio::error_code, Args...) &>
{
typedef R type(Args...) &;
};

template <typename R, typename... Args>
struct redirect_error_signature<R(const asio::error_code&, Args...) &>
{
typedef R type(Args...) &;
};

template <typename R, typename... Args>
struct redirect_error_signature<R(asio::error_code, Args...) &&>
{
typedef R type(Args...) &&;
};

template <typename R, typename... Args>
struct redirect_error_signature<R(const asio::error_code&, Args...) &&>
{
typedef R type(Args...) &&;
};

#  if defined(ASIO_HAS_NOEXCEPT_FUNCTION_TYPE)

template <typename R, typename... Args>
struct redirect_error_signature<
R(asio::error_code, Args...) noexcept>
{
typedef R type(Args...) & noexcept;
};

template <typename R, typename... Args>
struct redirect_error_signature<
R(const asio::error_code&, Args...) noexcept>
{
typedef R type(Args...) & noexcept;
};

template <typename R, typename... Args>
struct redirect_error_signature<
R(asio::error_code, Args...) & noexcept>
{
typedef R type(Args...) & noexcept;
};

template <typename R, typename... Args>
struct redirect_error_signature<
R(const asio::error_code&, Args...) & noexcept>
{
typedef R type(Args...) & noexcept;
};

template <typename R, typename... Args>
struct redirect_error_signature<
R(asio::error_code, Args...) && noexcept>
{
typedef R type(Args...) && noexcept;
};

template <typename R, typename... Args>
struct redirect_error_signature<
R(const asio::error_code&, Args...) && noexcept>
{
typedef R type(Args...) && noexcept;
};

#  endif 
# endif 
#else 

template <typename R>
struct redirect_error_signature<R(asio::error_code)>
{
typedef R type();
};

template <typename R>
struct redirect_error_signature<R(const asio::error_code&)>
{
typedef R type();
};

#define ASIO_PRIVATE_REDIRECT_ERROR_DEF(n) \
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(asio::error_code, ASIO_VARIADIC_TARGS(n))> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)); \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(const asio::error_code&, ASIO_VARIADIC_TARGS(n))> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)); \
}; \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_REDIRECT_ERROR_DEF)
#undef ASIO_PRIVATE_REDIRECT_ERROR_DEF

# if defined(ASIO_HAS_REF_QUALIFIED_FUNCTIONS)

template <typename R>
struct redirect_error_signature<R(asio::error_code) &>
{
typedef R type() &;
};

template <typename R>
struct redirect_error_signature<R(const asio::error_code&) &>
{
typedef R type() &;
};

template <typename R>
struct redirect_error_signature<R(asio::error_code) &&>
{
typedef R type() &&;
};

template <typename R>
struct redirect_error_signature<R(const asio::error_code&) &&>
{
typedef R type() &&;
};

#define ASIO_PRIVATE_REDIRECT_ERROR_DEF(n) \
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(asio::error_code, ASIO_VARIADIC_TARGS(n)) &> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) &; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(const asio::error_code&, ASIO_VARIADIC_TARGS(n)) &> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) &; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(asio::error_code, ASIO_VARIADIC_TARGS(n)) &&> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) &&; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(const asio::error_code&, ASIO_VARIADIC_TARGS(n)) &&> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) &&; \
}; \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_REDIRECT_ERROR_DEF)
#undef ASIO_PRIVATE_REDIRECT_ERROR_DEF

#  if defined(ASIO_HAS_NOEXCEPT_FUNCTION_TYPE)

template <typename R>
struct redirect_error_signature<
R(asio::error_code) noexcept>
{
typedef R type() noexcept;
};

template <typename R>
struct redirect_error_signature<
R(const asio::error_code&) noexcept>
{
typedef R type() noexcept;
};

template <typename R>
struct redirect_error_signature<
R(asio::error_code) & noexcept>
{
typedef R type() & noexcept;
};

template <typename R>
struct redirect_error_signature<
R(const asio::error_code&) & noexcept>
{
typedef R type() & noexcept;
};

template <typename R>
struct redirect_error_signature<
R(asio::error_code) && noexcept>
{
typedef R type() && noexcept;
};

template <typename R>
struct redirect_error_signature<
R(const asio::error_code&) && noexcept>
{
typedef R type() && noexcept;
};

#define ASIO_PRIVATE_REDIRECT_ERROR_DEF(n) \
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(asio::error_code, ASIO_VARIADIC_TARGS(n)) noexcept> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) noexcept; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(const asio::error_code&, \
ASIO_VARIADIC_TARGS(n)) noexcept> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) noexcept; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(asio::error_code, \
ASIO_VARIADIC_TARGS(n)) & noexcept> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) & noexcept; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(const asio::error_code&, \
ASIO_VARIADIC_TARGS(n)) & noexcept> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) & noexcept; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(asio::error_code, \
ASIO_VARIADIC_TARGS(n)) && noexcept> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) && noexcept; \
}; \
\
template <typename R, ASIO_VARIADIC_TPARAMS(n)> \
struct redirect_error_signature< \
R(const asio::error_code&, \
ASIO_VARIADIC_TARGS(n)) && noexcept> \
{ \
typedef R type(ASIO_VARIADIC_TARGS(n)) && noexcept; \
}; \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_REDIRECT_ERROR_DEF)
#undef ASIO_PRIVATE_REDIRECT_ERROR_DEF

#  endif 
# endif 
#endif 

} 

#if !defined(GENERATING_DOCUMENTATION)

template <typename CompletionToken, typename Signature>
struct async_result<redirect_error_t<CompletionToken>, Signature>
{
typedef typename async_result<CompletionToken,
typename detail::redirect_error_signature<Signature>::type>
::return_type return_type;

template <typename Initiation>
struct init_wrapper
{
template <typename Init>
init_wrapper(asio::error_code& ec, ASIO_MOVE_ARG(Init) init)
: ec_(ec),
initiation_(ASIO_MOVE_CAST(Init)(init))
{
}

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Handler, typename... Args>
void operator()(
ASIO_MOVE_ARG(Handler) handler,
ASIO_MOVE_ARG(Args)... args)
{
ASIO_MOVE_CAST(Initiation)(initiation_)(
detail::redirect_error_handler<
typename decay<Handler>::type>(
ec_, ASIO_MOVE_CAST(Handler)(handler)),
ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Handler>
void operator()(
ASIO_MOVE_ARG(Handler) handler)
{
ASIO_MOVE_CAST(Initiation)(initiation_)(
detail::redirect_error_handler<
typename decay<Handler>::type>(
ec_, ASIO_MOVE_CAST(Handler)(handler)));
}

#define ASIO_PRIVATE_INIT_WRAPPER_DEF(n) \
template <typename Handler, ASIO_VARIADIC_TPARAMS(n)> \
void operator()( \
ASIO_MOVE_ARG(Handler) handler, \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
ASIO_MOVE_CAST(Initiation)(initiation_)( \
detail::redirect_error_handler< \
typename decay<Handler>::type>( \
ec_, ASIO_MOVE_CAST(Handler)(handler)), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INIT_WRAPPER_DEF)
#undef ASIO_PRIVATE_INIT_WRAPPER_DEF

#endif 

asio::error_code& ec_;
Initiation initiation_;
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Initiation, typename RawCompletionToken, typename... Args>
static return_type initiate(
ASIO_MOVE_ARG(Initiation) initiation,
ASIO_MOVE_ARG(RawCompletionToken) token,
ASIO_MOVE_ARG(Args)... args)
{
return async_initiate<CompletionToken,
typename detail::redirect_error_signature<Signature>::type>(
init_wrapper<typename decay<Initiation>::type>(
token.ec_, ASIO_MOVE_CAST(Initiation)(initiation)),
token.token_, ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Initiation, typename RawCompletionToken>
static return_type initiate(
ASIO_MOVE_ARG(Initiation) initiation,
ASIO_MOVE_ARG(RawCompletionToken) token)
{
return async_initiate<CompletionToken,
typename detail::redirect_error_signature<Signature>::type>(
init_wrapper<typename decay<Initiation>::type>(
token.ec_, ASIO_MOVE_CAST(Initiation)(initiation)),
token.token_);
}

#define ASIO_PRIVATE_INITIATE_DEF(n) \
template <typename Initiation, typename RawCompletionToken, \
ASIO_VARIADIC_TPARAMS(n)> \
static return_type initiate( \
ASIO_MOVE_ARG(Initiation) initiation, \
ASIO_MOVE_ARG(RawCompletionToken) token, \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return async_initiate<CompletionToken, \
typename detail::redirect_error_signature<Signature>::type>( \
init_wrapper<typename decay<Initiation>::type>( \
token.ec_, ASIO_MOVE_CAST(Initiation)(initiation)), \
token.token_, ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INITIATE_DEF)
#undef ASIO_PRIVATE_INITIATE_DEF

#endif 
};

template <template <typename, typename> class Associator,
typename Handler, typename DefaultCandidate>
struct associator<Associator,
detail::redirect_error_handler<Handler>, DefaultCandidate>
: Associator<Handler, DefaultCandidate>
{
static typename Associator<Handler, DefaultCandidate>::type get(
const detail::redirect_error_handler<Handler>& h,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<Handler, DefaultCandidate>::get(h.handler_, c);
}
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
