
#ifndef BOOST_ASIO_BIND_EXECUTOR_HPP
#define BOOST_ASIO_BIND_EXECUTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/detail/variadic_templates.hpp>
#include <boost/asio/associated_executor.hpp>
#include <boost/asio/associated_allocator.hpp>
#include <boost/asio/async_result.hpp>
#include <boost/asio/execution/executor.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/is_executor.hpp>
#include <boost/asio/uses_executor.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
executor_binder_base(BOOST_ASIO_MOVE_ARG(E) e, BOOST_ASIO_MOVE_ARG(U) u)
: executor_(BOOST_ASIO_MOVE_CAST(E)(e)),
target_(executor_arg_t(), executor_, BOOST_ASIO_MOVE_CAST(U)(u))
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
executor_binder_base(BOOST_ASIO_MOVE_ARG(E) e, BOOST_ASIO_MOVE_ARG(U) u)
: executor_(BOOST_ASIO_MOVE_CAST(E)(e)),
target_(BOOST_ASIO_MOVE_CAST(U)(u))
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
BOOST_ASIO_MOVE_ARG(U) u)
: base_type(e, BOOST_ASIO_MOVE_CAST(U)(u))
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

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

executor_binder(executor_binder&& other)
: base_type(BOOST_ASIO_MOVE_CAST(executor_type)(other.get_executor()),
BOOST_ASIO_MOVE_CAST(T)(other.get()))
{
}

executor_binder(executor_arg_t, const executor_type& e,
executor_binder&& other)
: base_type(e, BOOST_ASIO_MOVE_CAST(T)(other.get()))
{
}

template <typename U, typename OtherExecutor>
executor_binder(executor_binder<U, OtherExecutor>&& other)
: base_type(BOOST_ASIO_MOVE_CAST(OtherExecutor)(other.get_executor()),
BOOST_ASIO_MOVE_CAST(U)(other.get()))
{
}

template <typename U, typename OtherExecutor>
executor_binder(executor_arg_t, const executor_type& e,
executor_binder<U, OtherExecutor>&& other)
: base_type(e, BOOST_ASIO_MOVE_CAST(U)(other.get()))
{
}

#endif 

~executor_binder()
{
}

target_type& get() BOOST_ASIO_NOEXCEPT
{
return this->target_;
}

const target_type& get() const BOOST_ASIO_NOEXCEPT
{
return this->target_;
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return this->executor_;
}

#if defined(GENERATING_DOCUMENTATION)

template <typename... Args> auto operator()(Args&& ...);
template <typename... Args> auto operator()(Args&& ...) const;

#elif defined(BOOST_ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... Args>
typename result_of<T(Args...)>::type operator()(
BOOST_ASIO_MOVE_ARG(Args)... args)
{
return this->target_(BOOST_ASIO_MOVE_CAST(Args)(args)...);
}

template <typename... Args>
typename result_of<T(Args...)>::type operator()(
BOOST_ASIO_MOVE_ARG(Args)... args) const
{
return this->target_(BOOST_ASIO_MOVE_CAST(Args)(args)...);
}

#elif defined(BOOST_ASIO_HAS_STD_TYPE_TRAITS) && !defined(_MSC_VER)

typename detail::executor_binder_result_of0<T>::type operator()()
{
return this->target_();
}

typename detail::executor_binder_result_of0<T>::type operator()() const
{
return this->target_();
}

#define BOOST_ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF(n) \
template <BOOST_ASIO_VARIADIC_TPARAMS(n)> \
typename result_of<T(BOOST_ASIO_VARIADIC_TARGS(n))>::type operator()( \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return this->target_(BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <BOOST_ASIO_VARIADIC_TPARAMS(n)> \
typename result_of<T(BOOST_ASIO_VARIADIC_TARGS(n))>::type operator()( \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
return this->target_(BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF)
#undef BOOST_ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF

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

#define BOOST_ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF(n) \
template <BOOST_ASIO_VARIADIC_TPARAMS(n)> \
result_type_or_void operator()( \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return this->target_(BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <BOOST_ASIO_VARIADIC_TPARAMS(n)> \
result_type_or_void operator()( \
BOOST_ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
return this->target_(BOOST_ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

BOOST_ASIO_VARIADIC_GENERATE(BOOST_ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF)
#undef BOOST_ASIO_PRIVATE_BIND_EXECUTOR_CALL_DEF

#endif 

private:
typedef detail::executor_binder_base<T, Executor,
uses_executor<T, Executor>::value> base_type;
};

template <typename Executor, typename T>
inline executor_binder<typename decay<T>::type, Executor>
bind_executor(const Executor& ex, BOOST_ASIO_MOVE_ARG(T) t,
typename enable_if<
is_executor<Executor>::value || execution::is_executor<Executor>::value
>::type* = 0)
{
return executor_binder<typename decay<T>::type, Executor>(
executor_arg_t(), ex, BOOST_ASIO_MOVE_CAST(T)(t));
}

template <typename ExecutionContext, typename T>
inline executor_binder<typename decay<T>::type,
typename ExecutionContext::executor_type>
bind_executor(ExecutionContext& ctx, BOOST_ASIO_MOVE_ARG(T) t,
typename enable_if<is_convertible<
ExecutionContext&, execution_context&>::value>::type* = 0)
{
return executor_binder<typename decay<T>::type,
typename ExecutionContext::executor_type>(
executor_arg_t(), ctx.get_executor(), BOOST_ASIO_MOVE_CAST(T)(t));
}

#if !defined(GENERATING_DOCUMENTATION)

template <typename T, typename Executor>
struct uses_executor<executor_binder<T, Executor>, Executor>
: true_type {};

template <typename T, typename Executor, typename Signature>
class async_result<executor_binder<T, Executor>, Signature>
{
public:
typedef executor_binder<
typename async_result<T, Signature>::completion_handler_type, Executor>
completion_handler_type;

typedef typename async_result<T, Signature>::return_type return_type;

explicit async_result(executor_binder<T, Executor>& b)
: target_(b.get())
{
}

return_type get()
{
return target_.get();
}

private:
async_result(const async_result&) BOOST_ASIO_DELETED;
async_result& operator=(const async_result&) BOOST_ASIO_DELETED;

async_result<T, Signature> target_;
};

template <typename T, typename Executor, typename Allocator>
struct associated_allocator<executor_binder<T, Executor>, Allocator>
{
typedef typename associated_allocator<T, Allocator>::type type;

static type get(const executor_binder<T, Executor>& b,
const Allocator& a = Allocator()) BOOST_ASIO_NOEXCEPT
{
return associated_allocator<T, Allocator>::get(b.get(), a);
}
};

template <typename T, typename Executor, typename Executor1>
struct associated_executor<executor_binder<T, Executor>, Executor1>
{
typedef Executor type;

static type get(const executor_binder<T, Executor>& b,
const Executor1& = Executor1()) BOOST_ASIO_NOEXCEPT
{
return b.get_executor();
}
};

#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
