
#ifndef ASIO_BIND_EXECUTOR_HPP
#define ASIO_BIND_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/associated_executor.hpp"
#include "asio/associator.hpp"
#include "asio/async_result.hpp"
#include "asio/execution/executor.hpp"
#include "asio/execution_context.hpp"
#include "asio/is_executor.hpp"
#include "asio/uses_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {


template <typename T, typename = void>
struct executor_binder_result_type
{
protected:
typedef void result_type_or_void;
};

template <typename T>
struct executor_binder_result_type<T,
typename void_type<typename T::result_type>::type>
{
typedef typename T::result_type result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R>
struct executor_binder_result_type<R(*)()>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R>
struct executor_binder_result_type<R(&)()>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1>
struct executor_binder_result_type<R(*)(A1)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1>
struct executor_binder_result_type<R(&)(A1)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1, typename A2>
struct executor_binder_result_type<R(*)(A1, A2)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1, typename A2>
struct executor_binder_result_type<R(&)(A1, A2)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};


template <typename T, typename = void>
struct executor_binder_argument_type {};

template <typename T>
struct executor_binder_argument_type<T,
typename void_type<typename T::argument_type>::type>
{
typedef typename T::argument_type argument_type;
};

template <typename R, typename A1>
struct executor_binder_argument_type<R(*)(A1)>
{
typedef A1 argument_type;
};

template <typename R, typename A1>
struct executor_binder_argument_type<R(&)(A1)>
{
typedef A1 argument_type;
};


template <typename T, typename = void>
struct executor_binder_argument_types {};

template <typename T>
struct executor_binder_argument_types<T,
typename void_type<typename T::first_argument_type>::type>
{
typedef typename T::first_argument_type first_argument_type;
typedef typename T::second_argument_type second_argument_type;
};

template <typename R, typename A1, typename A2>
struct executor_binder_argument_type<R(*)(A1, A2)>
{
typedef A1 first_argument_type;
typedef A2 second_argument_type;
};

template <typename R, typename A1, typename A2>
struct executor_binder_argument_type<R(&)(A1, A2)>
{
typedef A1 first_argument_type;
typedef A2 second_argument_type;
};


template <typename T, typename Executor, bool UsesExecutor>
class executor_binder_base;

template <typename T, typename Executor>
class executor_binder_base<T, Executor, true>
{
protected:
template <typename E, typename U>
executor_binder_base(ASIO_MOVE_ARG(E) e, ASIO_MOVE_ARG(U) u)
: executor_(ASIO_MOVE_CAST(E)(e)),
target_(executor_arg_t(), executor_, ASIO_MOVE_CAST(U)(u))
{
}

Executor executor_;
T target_;
};

template <typename T, typename Executor>
class executor_binder_base<T, Executor, false>
{
protected:
template <typename E, typename U>
executor_binder_base(ASIO_MOVE_ARG(E) e, ASIO_MOVE_ARG(U) u)
: executor_(ASIO_MOVE_CAST(E)(e)),
target_(ASIO_MOVE_CAST(U)(u))
{
}

Executor executor_;
T target_;
};


template <typename T, typename = void>
struct executor_binder_result_of0
{
typedef void type;
};

template <typename T>
struct executor_binder_result_of0<T,
typename void_type<typename result_of<T()>::type>::type>
{
typedef typename result_of<T()>::type type;
};

} 

template <typename T, typename Executor>
class executor_binder
#if !defined(GENERATING_DOCUMENTATION)
: public detail::executor_binder_result_type<T>,
public detail::executor_binder_argument_type<T>,
public detail::executor_binder_argument_types<T>,
private detail::executor_binder_base<
T, Executor, uses_executor<T, Executor>::value>
#endif 
{
public:
typedef T target_type;

typedef Executor executor_type;

#if defined(GENERATING_DOCUMENTATION)

typedef see_below result_type;


typedef see_below argument_type;


typedef see_below first_argument_type;


typedef see_below second_argument_type;
#endif 


template <typename U>
executor_binder(executor_arg_t, const executor_type& e,
ASIO_MOVE_ARG(U) u)
: base_type(e, ASIO_MOVE_CAST(U)(u))
{
}

executor_binder(const executor_binder& other)
: base_type(other.get_executor(), other.get())
{
}

executor_binder(executor_arg_t, const executor_type& e,
const executor_binder& other)
: base_type(e, other.get())
{
}


template <typename U, typename OtherExecutor>
executor_binder(const executor_binder<U, OtherExecutor>& other)
: base_type(other.get_executor(), other.get())
{
}


template <typename U, typename OtherExecutor>
executor_binder(executor_arg_t, const executor_type& e,
const executor_binder<U, OtherExecutor>& other)
: base_type(e, other.get())
{
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

executor_binder(executor_binder&& other)
: base_type(ASIO_MOVE_CAST(executor_type)(other.get_executor()),
ASIO_MOVE_CAST(T)(other.get()))
{
}

executor_binder(executor_arg_t, const executor_type& e,
executor_binder&& other)
: base_type(e, ASIO_MOVE_CAST(T)(other.get()))
{
}

template <typename U, typename OtherExecutor>
executor_binder(executor_binder<U, OtherExecutor>&& other)
: base_type(ASIO_MOVE_CAST(OtherExecutor)(other.get_executor()),
ASIO_MOVE_CAST(U)(other.get()))
{
}

template <typename U, typename OtherExecutor>
executor_binder(executor_arg_t, const executor_type& e,
executor_binder<U, OtherExecutor>&& other)
: base_type(e, ASIO_MOVE_CAST(U)(other.get()))
{
}

#endif 

~executor_binder()
{
}

target_type& get() ASIO_NOEXCEPT
{
return this->target_;
}

const target_type& get() const ASIO_NOEXCEPT
{
return this->target_;
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return this->executor_;
}

#if defined(GENERATING_DOCUMENTATION)

template <typename... Args> auto operator()(Args&& ...);
template <typename... Args> auto operator()(Args&& ...) const;

#elif defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... Args>
typename result_of<T(Args...)>::type operator()(
ASIO_MOVE_ARG(Args)... args)
{
return this->target_(ASIO_MOVE_CAST(Args)(args)...);
}

template <typename... Args>
typename result_of<T(Args...)>::type operator()(
ASIO_MOVE_ARG(Args)... args) const
{
return this->target_(ASIO_MOVE_CAST(Args)(args)...);
}

#elif defined(ASIO_HAS_STD_TYPE_TRAITS) && !defined(_MSC_VER)

typename detail::executor_binder_result_of0<T>::type operator()()
{
return this->target_();
}

typename detail::executor_binder_result_of0<T>::type operator()() const
{
return this->target_();
}

#define ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
typename result_of<T(ASIO_VARIADIC_TARGS(n))>::type operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return this->target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <ASIO_VARIADIC_TPARAMS(n)> \
typename result_of<T(ASIO_VARIADIC_TARGS(n))>::type operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
return this->target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF)
#undef ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF

#else 

typedef typename detail::executor_binder_result_type<T>::result_type_or_void
result_type_or_void;

result_type_or_void operator()()
{
return this->target_();
}

result_type_or_void operator()() const
{
return this->target_();
}

#define ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
result_type_or_void operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return this->target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <ASIO_VARIADIC_TPARAMS(n)> \
result_type_or_void operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
return this->target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF)
#undef ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF

#endif 

private:
typedef detail::executor_binder_base<T, Executor,
uses_executor<T, Executor>::value> base_type;
};

template <typename Executor, typename T>
inline executor_binder<typename decay<T>::type, Executor>
bind_executor(const Executor& ex, ASIO_MOVE_ARG(T) t,
typename constraint<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type = 0)
{
return executor_binder<typename decay<T>::type, Executor>(
executor_arg_t(), ex, ASIO_MOVE_CAST(T)(t));
}

template <typename ExecutionContext, typename T>
inline executor_binder<typename decay<T>::type,
typename ExecutionContext::executor_type>
bind_executor(ExecutionContext& ctx, ASIO_MOVE_ARG(T) t,
typename constraint<is_convertible<
ExecutionContext&, execution_context&>::value>::type = 0)
{
return executor_binder<typename decay<T>::type,
typename ExecutionContext::executor_type>(
executor_arg_t(), ctx.get_executor(), ASIO_MOVE_CAST(T)(t));
}

#if !defined(GENERATING_DOCUMENTATION)

template <typename T, typename Executor>
struct uses_executor<executor_binder<T, Executor>, Executor>
: true_type {};

namespace detail {

template <typename TargetAsyncResult, typename Executor, typename = void>
struct executor_binder_async_result_completion_handler_type
{
};

template <typename TargetAsyncResult, typename Executor>
struct executor_binder_async_result_completion_handler_type<
TargetAsyncResult, Executor,
typename void_type<
typename TargetAsyncResult::completion_handler_type
>::type>
{
typedef executor_binder<
typename TargetAsyncResult::completion_handler_type, Executor>
completion_handler_type;
};

template <typename TargetAsyncResult, typename = void>
struct executor_binder_async_result_return_type
{
};

template <typename TargetAsyncResult>
struct executor_binder_async_result_return_type<
TargetAsyncResult,
typename void_type<
typename TargetAsyncResult::return_type
>::type>
{
typedef typename TargetAsyncResult::return_type return_type;
};

} 

template <typename T, typename Executor, typename Signature>
class async_result<executor_binder<T, Executor>, Signature> :
public detail::executor_binder_async_result_completion_handler_type<
async_result<T, Signature>, Executor>,
public detail::executor_binder_async_result_return_type<
async_result<T, Signature> >
{
public:
explicit async_result(executor_binder<T, Executor>& b)
: target_(b.get())
{
}

typename async_result<T, Signature>::return_type get()
{
return target_.get();
}

template <typename Initiation>
struct init_wrapper
{
template <typename Init>
init_wrapper(const Executor& ex, ASIO_MOVE_ARG(Init) init)
: ex_(ex),
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
executor_binder<typename decay<Handler>::type, Executor>(
executor_arg_t(), ex_, ASIO_MOVE_CAST(Handler)(handler)),
ASIO_MOVE_CAST(Args)(args)...);
}

template <typename Handler, typename... Args>
void operator()(
ASIO_MOVE_ARG(Handler) handler,
ASIO_MOVE_ARG(Args)... args) const
{
initiation_(
executor_binder<typename decay<Handler>::type, Executor>(
executor_arg_t(), ex_, ASIO_MOVE_CAST(Handler)(handler)),
ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Handler>
void operator()(
ASIO_MOVE_ARG(Handler) handler)
{
ASIO_MOVE_CAST(Initiation)(initiation_)(
executor_binder<typename decay<Handler>::type, Executor>(
executor_arg_t(), ex_, ASIO_MOVE_CAST(Handler)(handler)));
}

template <typename Handler>
void operator()(
ASIO_MOVE_ARG(Handler) handler) const
{
initiation_(
executor_binder<typename decay<Handler>::type, Executor>(
executor_arg_t(), ex_, ASIO_MOVE_CAST(Handler)(handler)));
}

#define ASIO_PRIVATE_INIT_WRAPPER_DEF(n) \
template <typename Handler, ASIO_VARIADIC_TPARAMS(n)> \
void operator()( \
ASIO_MOVE_ARG(Handler) handler, \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
ASIO_MOVE_CAST(Initiation)(initiation_)( \
executor_binder<typename decay<Handler>::type, Executor>( \
executor_arg_t(), ex_, ASIO_MOVE_CAST(Handler)(handler)), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <typename Handler, ASIO_VARIADIC_TPARAMS(n)> \
void operator()( \
ASIO_MOVE_ARG(Handler) handler, \
ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
initiation_( \
executor_binder<typename decay<Handler>::type, Executor>( \
executor_arg_t(), ex_, ASIO_MOVE_CAST(Handler)(handler)), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INIT_WRAPPER_DEF)
#undef ASIO_PRIVATE_INIT_WRAPPER_DEF

#endif 

Executor ex_;
Initiation initiation_;
};

#if defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename Initiation, typename RawCompletionToken, typename... Args>
static ASIO_INITFN_DEDUCED_RESULT_TYPE(T, Signature,
(async_result<T, Signature>::initiate(
declval<init_wrapper<typename decay<Initiation>::type> >(),
declval<T>(), declval<ASIO_MOVE_ARG(Args)>()...)))
initiate(
ASIO_MOVE_ARG(Initiation) initiation,
ASIO_MOVE_ARG(RawCompletionToken) token,
ASIO_MOVE_ARG(Args)... args)
{
return async_initiate<T, Signature>(
init_wrapper<typename decay<Initiation>::type>(
token.get_executor(), ASIO_MOVE_CAST(Initiation)(initiation)),
token.get(), ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Initiation, typename RawCompletionToken>
static ASIO_INITFN_DEDUCED_RESULT_TYPE(T, Signature,
(async_result<T, Signature>::initiate(
declval<init_wrapper<typename decay<Initiation>::type> >(),
declval<T>())))
initiate(
ASIO_MOVE_ARG(Initiation) initiation,
ASIO_MOVE_ARG(RawCompletionToken) token)
{
return async_initiate<T, Signature>(
init_wrapper<typename decay<Initiation>::type>(
token.get_executor(), ASIO_MOVE_CAST(Initiation)(initiation)),
token.get());
}

#define ASIO_PRIVATE_INITIATE_DEF(n) \
template <typename Initiation, typename RawCompletionToken, \
ASIO_VARIADIC_TPARAMS(n)> \
static ASIO_INITFN_DEDUCED_RESULT_TYPE(T, Signature, \
(async_result<T, Signature>::initiate( \
declval<init_wrapper<typename decay<Initiation>::type> >(), \
declval<T>(), ASIO_VARIADIC_MOVE_DECLVAL(n)))) \
initiate( \
ASIO_MOVE_ARG(Initiation) initiation, \
ASIO_MOVE_ARG(RawCompletionToken) token, \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return async_initiate<T, Signature>( \
init_wrapper<typename decay<Initiation>::type>( \
token.get_executor(), ASIO_MOVE_CAST(Initiation)(initiation)), \
token.get(), ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INITIATE_DEF)
#undef ASIO_PRIVATE_INITIATE_DEF

#endif 

private:
async_result(const async_result&) ASIO_DELETED;
async_result& operator=(const async_result&) ASIO_DELETED;

async_result<T, Signature> target_;
};

template <template <typename, typename> class Associator,
typename T, typename Executor, typename DefaultCandidate>
struct associator<Associator, executor_binder<T, Executor>, DefaultCandidate>
{
typedef typename Associator<T, DefaultCandidate>::type type;

static type get(const executor_binder<T, Executor>& b,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<T, DefaultCandidate>::get(b.get(), c);
}
};

template <typename T, typename Executor, typename Executor1>
struct associated_executor<executor_binder<T, Executor>, Executor1>
{
typedef Executor type;

static type get(const executor_binder<T, Executor>& b,
const Executor1& = Executor1()) ASIO_NOEXCEPT
{
return b.get_executor();
}
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
