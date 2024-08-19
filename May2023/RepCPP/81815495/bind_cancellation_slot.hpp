
#ifndef ASIO_BIND_CANCELLATION_SLOT_HPP
#define ASIO_BIND_CANCELLATION_SLOT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/detail/variadic_templates.hpp"
#include "asio/associated_cancellation_slot.hpp"
#include "asio/associator.hpp"
#include "asio/async_result.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {


template <typename T, typename = void>
struct cancellation_slot_binder_result_type
{
protected:
typedef void result_type_or_void;
};

template <typename T>
struct cancellation_slot_binder_result_type<T,
typename void_type<typename T::result_type>::type>
{
typedef typename T::result_type result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R>
struct cancellation_slot_binder_result_type<R(*)()>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R>
struct cancellation_slot_binder_result_type<R(&)()>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1>
struct cancellation_slot_binder_result_type<R(*)(A1)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1>
struct cancellation_slot_binder_result_type<R(&)(A1)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1, typename A2>
struct cancellation_slot_binder_result_type<R(*)(A1, A2)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};

template <typename R, typename A1, typename A2>
struct cancellation_slot_binder_result_type<R(&)(A1, A2)>
{
typedef R result_type;
protected:
typedef result_type result_type_or_void;
};


template <typename T, typename = void>
struct cancellation_slot_binder_argument_type {};

template <typename T>
struct cancellation_slot_binder_argument_type<T,
typename void_type<typename T::argument_type>::type>
{
typedef typename T::argument_type argument_type;
};

template <typename R, typename A1>
struct cancellation_slot_binder_argument_type<R(*)(A1)>
{
typedef A1 argument_type;
};

template <typename R, typename A1>
struct cancellation_slot_binder_argument_type<R(&)(A1)>
{
typedef A1 argument_type;
};


template <typename T, typename = void>
struct cancellation_slot_binder_argument_types {};

template <typename T>
struct cancellation_slot_binder_argument_types<T,
typename void_type<typename T::first_argument_type>::type>
{
typedef typename T::first_argument_type first_argument_type;
typedef typename T::second_argument_type second_argument_type;
};

template <typename R, typename A1, typename A2>
struct cancellation_slot_binder_argument_type<R(*)(A1, A2)>
{
typedef A1 first_argument_type;
typedef A2 second_argument_type;
};

template <typename R, typename A1, typename A2>
struct cancellation_slot_binder_argument_type<R(&)(A1, A2)>
{
typedef A1 first_argument_type;
typedef A2 second_argument_type;
};


template <typename T, typename = void>
struct cancellation_slot_binder_result_of0
{
typedef void type;
};

template <typename T>
struct cancellation_slot_binder_result_of0<T,
typename void_type<typename result_of<T()>::type>::type>
{
typedef typename result_of<T()>::type type;
};

} 

template <typename T, typename CancellationSlot>
class cancellation_slot_binder
#if !defined(GENERATING_DOCUMENTATION)
: public detail::cancellation_slot_binder_result_type<T>,
public detail::cancellation_slot_binder_argument_type<T>,
public detail::cancellation_slot_binder_argument_types<T>
#endif 
{
public:
typedef T target_type;

typedef CancellationSlot cancellation_slot_type;

#if defined(GENERATING_DOCUMENTATION)

typedef see_below result_type;


typedef see_below argument_type;


typedef see_below first_argument_type;


typedef see_below second_argument_type;
#endif 


template <typename U>
cancellation_slot_binder(const cancellation_slot_type& s,
ASIO_MOVE_ARG(U) u)
: slot_(s),
target_(ASIO_MOVE_CAST(U)(u))
{
}

cancellation_slot_binder(const cancellation_slot_binder& other)
: slot_(other.get_cancellation_slot()),
target_(other.get())
{
}

cancellation_slot_binder(const cancellation_slot_type& s,
const cancellation_slot_binder& other)
: slot_(s),
target_(other.get())
{
}


template <typename U, typename OtherCancellationSlot>
cancellation_slot_binder(
const cancellation_slot_binder<U, OtherCancellationSlot>& other)
: slot_(other.get_cancellation_slot()),
target_(other.get())
{
}


template <typename U, typename OtherCancellationSlot>
cancellation_slot_binder(const cancellation_slot_type& s,
const cancellation_slot_binder<U, OtherCancellationSlot>& other)
: slot_(s),
target_(other.get())
{
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

cancellation_slot_binder(cancellation_slot_binder&& other)
: slot_(ASIO_MOVE_CAST(cancellation_slot_type)(
other.get_cancellation_slot())),
target_(ASIO_MOVE_CAST(T)(other.get()))
{
}

cancellation_slot_binder(const cancellation_slot_type& s,
cancellation_slot_binder&& other)
: slot_(s),
target_(ASIO_MOVE_CAST(T)(other.get()))
{
}

template <typename U, typename OtherCancellationSlot>
cancellation_slot_binder(
cancellation_slot_binder<U, OtherCancellationSlot>&& other)
: slot_(ASIO_MOVE_CAST(OtherCancellationSlot)(
other.get_cancellation_slot())),
target_(ASIO_MOVE_CAST(U)(other.get()))
{
}

template <typename U, typename OtherCancellationSlot>
cancellation_slot_binder(const cancellation_slot_type& s,
cancellation_slot_binder<U, OtherCancellationSlot>&& other)
: slot_(s),
target_(ASIO_MOVE_CAST(U)(other.get()))
{
}

#endif 

~cancellation_slot_binder()
{
}

target_type& get() ASIO_NOEXCEPT
{
return target_;
}

const target_type& get() const ASIO_NOEXCEPT
{
return target_;
}

cancellation_slot_type get_cancellation_slot() const ASIO_NOEXCEPT
{
return slot_;
}

#if defined(GENERATING_DOCUMENTATION)

template <typename... Args> auto operator()(Args&& ...);
template <typename... Args> auto operator()(Args&& ...) const;

#elif defined(ASIO_HAS_VARIADIC_TEMPLATES)

template <typename... Args>
typename result_of<T(Args...)>::type operator()(
ASIO_MOVE_ARG(Args)... args)
{
return target_(ASIO_MOVE_CAST(Args)(args)...);
}

template <typename... Args>
typename result_of<T(Args...)>::type operator()(
ASIO_MOVE_ARG(Args)... args) const
{
return target_(ASIO_MOVE_CAST(Args)(args)...);
}

#elif defined(ASIO_HAS_STD_TYPE_TRAITS) && !defined(_MSC_VER)

typename detail::cancellation_slot_binder_result_of0<T>::type operator()()
{
return target_();
}

typename detail::cancellation_slot_binder_result_of0<T>::type
operator()() const
{
return target_();
}

#define ASIO_PRIVATE_BINDER_CALL_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
typename result_of<T(ASIO_VARIADIC_TARGS(n))>::type operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <ASIO_VARIADIC_TPARAMS(n)> \
typename result_of<T(ASIO_VARIADIC_TARGS(n))>::type operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
return target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_BINDER_CALL_DEF)
#undef ASIO_PRIVATE_BINDER_CALL_DEF

#else 

typedef typename detail::cancellation_slot_binder_result_type<
T>::result_type_or_void result_type_or_void;

result_type_or_void operator()()
{
return target_();
}

result_type_or_void operator()() const
{
return target_();
}

#define ASIO_PRIVATE_BINDER_CALL_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
result_type_or_void operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
return target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <ASIO_VARIADIC_TPARAMS(n)> \
result_type_or_void operator()( \
ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
return target_(ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_BINDER_CALL_DEF)
#undef ASIO_PRIVATE_BINDER_CALL_DEF

#endif 

private:
CancellationSlot slot_;
T target_;
};

template <typename CancellationSlot, typename T>
inline cancellation_slot_binder<typename decay<T>::type, CancellationSlot>
bind_cancellation_slot(const CancellationSlot& s, ASIO_MOVE_ARG(T) t)
{
return cancellation_slot_binder<
typename decay<T>::type, CancellationSlot>(
s, ASIO_MOVE_CAST(T)(t));
}

#if !defined(GENERATING_DOCUMENTATION)

namespace detail {

template <typename TargetAsyncResult,
typename CancellationSlot, typename = void>
struct cancellation_slot_binder_async_result_completion_handler_type
{
};

template <typename TargetAsyncResult, typename CancellationSlot>
struct cancellation_slot_binder_async_result_completion_handler_type<
TargetAsyncResult, CancellationSlot,
typename void_type<
typename TargetAsyncResult::completion_handler_type
>::type>
{
typedef cancellation_slot_binder<
typename TargetAsyncResult::completion_handler_type, CancellationSlot>
completion_handler_type;
};

template <typename TargetAsyncResult, typename = void>
struct cancellation_slot_binder_async_result_return_type
{
};

template <typename TargetAsyncResult>
struct cancellation_slot_binder_async_result_return_type<
TargetAsyncResult,
typename void_type<
typename TargetAsyncResult::return_type
>::type>
{
typedef typename TargetAsyncResult::return_type return_type;
};

} 

template <typename T, typename CancellationSlot, typename Signature>
class async_result<cancellation_slot_binder<T, CancellationSlot>, Signature> :
public detail::cancellation_slot_binder_async_result_completion_handler_type<
async_result<T, Signature>, CancellationSlot>,
public detail::cancellation_slot_binder_async_result_return_type<
async_result<T, Signature> >
{
public:
explicit async_result(cancellation_slot_binder<T, CancellationSlot>& b)
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
init_wrapper(const CancellationSlot& slot, ASIO_MOVE_ARG(Init) init)
: slot_(slot),
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
cancellation_slot_binder<
typename decay<Handler>::type, CancellationSlot>(
slot_, ASIO_MOVE_CAST(Handler)(handler)),
ASIO_MOVE_CAST(Args)(args)...);
}

template <typename Handler, typename... Args>
void operator()(
ASIO_MOVE_ARG(Handler) handler,
ASIO_MOVE_ARG(Args)... args) const
{
initiation_(
cancellation_slot_binder<
typename decay<Handler>::type, CancellationSlot>(
slot_, ASIO_MOVE_CAST(Handler)(handler)),
ASIO_MOVE_CAST(Args)(args)...);
}

#else 

template <typename Handler>
void operator()(
ASIO_MOVE_ARG(Handler) handler)
{
ASIO_MOVE_CAST(Initiation)(initiation_)(
cancellation_slot_binder<
typename decay<Handler>::type, CancellationSlot>(
slot_, ASIO_MOVE_CAST(Handler)(handler)));
}

template <typename Handler>
void operator()(
ASIO_MOVE_ARG(Handler) handler) const
{
initiation_(
cancellation_slot_binder<
typename decay<Handler>::type, CancellationSlot>(
slot_, ASIO_MOVE_CAST(Handler)(handler)));
}

#define ASIO_PRIVATE_INIT_WRAPPER_DEF(n) \
template <typename Handler, ASIO_VARIADIC_TPARAMS(n)> \
void operator()( \
ASIO_MOVE_ARG(Handler) handler, \
ASIO_VARIADIC_MOVE_PARAMS(n)) \
{ \
ASIO_MOVE_CAST(Initiation)(initiation_)( \
cancellation_slot_binder< \
typename decay<Handler>::type, CancellationSlot>( \
slot_, ASIO_MOVE_CAST(Handler)(handler)), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \
\
template <typename Handler, ASIO_VARIADIC_TPARAMS(n)> \
void operator()( \
ASIO_MOVE_ARG(Handler) handler, \
ASIO_VARIADIC_MOVE_PARAMS(n)) const \
{ \
initiation_( \
cancellation_slot_binder< \
typename decay<Handler>::type, CancellationSlot>( \
slot_, ASIO_MOVE_CAST(Handler)(handler)), \
ASIO_VARIADIC_MOVE_ARGS(n)); \
} \

ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_INIT_WRAPPER_DEF)
#undef ASIO_PRIVATE_INIT_WRAPPER_DEF

#endif 

CancellationSlot slot_;
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
token.get_cancellation_slot(),
ASIO_MOVE_CAST(Initiation)(initiation)),
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
token.get_cancellation_slot(),
ASIO_MOVE_CAST(Initiation)(initiation)),
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
token.get_cancellation_slot(), \
ASIO_MOVE_CAST(Initiation)(initiation)), \
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
typename T, typename CancellationSlot, typename DefaultCandidate>
struct associator<Associator,
cancellation_slot_binder<T, CancellationSlot>,
DefaultCandidate>
{
typedef typename Associator<T, DefaultCandidate>::type type;

static type get(const cancellation_slot_binder<T, CancellationSlot>& b,
const DefaultCandidate& c = DefaultCandidate()) ASIO_NOEXCEPT
{
return Associator<T, DefaultCandidate>::get(b.get(), c);
}
};

template <typename T, typename CancellationSlot, typename CancellationSlot1>
struct associated_cancellation_slot<
cancellation_slot_binder<T, CancellationSlot>,
CancellationSlot1>
{
typedef CancellationSlot type;

static type get(const cancellation_slot_binder<T, CancellationSlot>& b,
const CancellationSlot1& = CancellationSlot1()) ASIO_NOEXCEPT
{
return b.get_cancellation_slot();
}
};

#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
