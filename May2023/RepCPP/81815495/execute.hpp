
#ifndef ASIO_EXECUTION_EXECUTE_HPP
#define ASIO_EXECUTION_EXECUTE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/execution/detail/as_invocable.hpp"
#include "asio/execution/detail/as_receiver.hpp"
#include "asio/traits/execute_member.hpp"
#include "asio/traits/execute_free.hpp"

#include "asio/detail/push_options.hpp"

#if defined(GENERATING_DOCUMENTATION)

namespace asio {
namespace execution {


inline constexpr unspecified execute = unspecified;


template <typename T, typename F>
struct can_execute :
integral_constant<bool, automatically_determined>
{
};

} 
} 

#else 

namespace asio {
namespace execution {

template <typename T, typename R>
struct is_sender_to;

namespace detail {

template <typename S, typename R>
void submit_helper(ASIO_MOVE_ARG(S) s, ASIO_MOVE_ARG(R) r);

} 
} 
} 
namespace asio_execution_execute_fn {

using asio::conditional;
using asio::decay;
using asio::declval;
using asio::enable_if;
using asio::execution::detail::as_receiver;
using asio::execution::detail::is_as_invocable;
using asio::execution::is_sender_to;
using asio::false_type;
using asio::result_of;
using asio::traits::execute_free;
using asio::traits::execute_member;
using asio::true_type;
using asio::void_type;

void execute();

enum overload_type
{
call_member,
call_free,
adapter,
ill_formed
};

template <typename Impl, typename T, typename F, typename = void,
typename = void, typename = void, typename = void, typename = void>
struct call_traits
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = ill_formed);
};

template <typename Impl, typename T, typename F>
struct call_traits<Impl, T, void(F),
typename enable_if<
execute_member<typename Impl::template proxy<T>::type, F>::is_valid
>::type> :
execute_member<typename Impl::template proxy<T>::type, F>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = call_member);
};

template <typename Impl, typename T, typename F>
struct call_traits<Impl, T, void(F),
typename enable_if<
!execute_member<typename Impl::template proxy<T>, F>::is_valid
>::type,
typename enable_if<
execute_free<T, F>::is_valid
>::type> :
execute_free<T, F>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

template <typename Impl, typename T, typename F>
struct call_traits<Impl, T, void(F),
typename enable_if<
!execute_member<typename Impl::template proxy<T>::type, F>::is_valid
>::type,
typename enable_if<
!execute_free<T, F>::is_valid
>::type,
typename void_type<
typename result_of<typename decay<F>::type&()>::type
>::type,
typename enable_if<
!is_as_invocable<typename decay<F>::type>::value
>::type,
typename enable_if<
is_sender_to<T, as_receiver<typename decay<F>::type, T> >::value
>::type>
{
ASIO_STATIC_CONSTEXPR(overload_type, overload = adapter);
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

struct impl
{
template <typename T>
struct proxy
{
#if defined(ASIO_HAS_DEDUCED_EXECUTE_MEMBER_TRAIT)
struct type
{
template <typename F>
auto execute(ASIO_MOVE_ARG(F) f)
noexcept(
noexcept(
declval<typename conditional<true, T, F>::type>().execute(
ASIO_MOVE_CAST(F)(f))
)
)
-> decltype(
declval<typename conditional<true, T, F>::type>().execute(
ASIO_MOVE_CAST(F)(f))
);
};
#else 
typedef T type;
#endif 
};

template <typename T, typename F>
ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(F)>::overload == call_member,
typename call_traits<impl, T, void(F)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T) t,
ASIO_MOVE_ARG(F) f) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(F)>::is_noexcept))
{
return ASIO_MOVE_CAST(T)(t).execute(ASIO_MOVE_CAST(F)(f));
}

template <typename T, typename F>
ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(F)>::overload == call_free,
typename call_traits<impl, T, void(F)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T) t,
ASIO_MOVE_ARG(F) f) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(F)>::is_noexcept))
{
return execute(ASIO_MOVE_CAST(T)(t), ASIO_MOVE_CAST(F)(f));
}

template <typename T, typename F>
ASIO_CONSTEXPR typename enable_if<
call_traits<impl, T, void(F)>::overload == adapter,
typename call_traits<impl, T, void(F)>::result_type
>::type
operator()(
ASIO_MOVE_ARG(T) t,
ASIO_MOVE_ARG(F) f) const
ASIO_NOEXCEPT_IF((
call_traits<impl, T, void(F)>::is_noexcept))
{
return asio::execution::detail::submit_helper(
ASIO_MOVE_CAST(T)(t),
as_receiver<typename decay<F>::type, T>(
ASIO_MOVE_CAST(F)(f), 0));
}
};

template <typename T = impl>
struct static_instance
{
static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} 
namespace asio {
namespace execution {
namespace {

static ASIO_CONSTEXPR const asio_execution_execute_fn::impl&
execute = asio_execution_execute_fn::static_instance<>::instance;

} 

typedef asio_execution_execute_fn::impl execute_t;

template <typename T, typename F>
struct can_execute :
integral_constant<bool,
asio_execution_execute_fn::call_traits<
execute_t, T, void(F)>::overload !=
asio_execution_execute_fn::ill_formed>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename F>
constexpr bool can_execute_v = can_execute<T, F>::value;

#endif 

} 
} 

#endif 

#include "asio/detail/pop_options.hpp"

#endif 
