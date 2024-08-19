
#ifndef BOOST_ASIO_EXECUTION_BULK_EXECUTE_HPP
#define BOOST_ASIO_EXECUTION_BULK_EXECUTE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/execution/bulk_guarantee.hpp>
#include <boost/asio/execution/detail/bulk_sender.hpp>
#include <boost/asio/execution/executor.hpp>
#include <boost/asio/execution/sender.hpp>
#include <boost/asio/traits/bulk_execute_member.hpp>
#include <boost/asio/traits/bulk_execute_free.hpp>

#include <boost/asio/detail/push_options.hpp>

#if defined(GENERATING_DOCUMENTATION)

namespace boost {
namespace asio {
namespace execution {


inline constexpr unspecified bulk_execute = unspecified;


template <typename S, typename F, typename N>
struct can_bulk_execute :
integral_constant<bool, automatically_determined>
{
};

} 
} 
} 

#else 

namespace asio_execution_bulk_execute_fn {

using boost::asio::declval;
using boost::asio::enable_if;
using boost::asio::execution::bulk_guarantee_t;
using boost::asio::execution::detail::bulk_sender;
using boost::asio::execution::executor_index;
using boost::asio::execution::is_sender;
using boost::asio::is_convertible;
using boost::asio::is_same;
using boost::asio::remove_cvref;
using boost::asio::result_of;
using boost::asio::traits::bulk_execute_free;
using boost::asio::traits::bulk_execute_member;
using boost::asio::traits::static_require;

void bulk_execute();

enum overload_type
{
call_member,
call_free,
adapter,
ill_formed
};

template <typename S, typename Args, typename = void>
struct call_traits
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = ill_formed);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

template <typename S, typename F, typename N>
struct call_traits<S, void(F, N),
typename enable_if<
(
is_convertible<N, std::size_t>::value
&&
bulk_execute_member<S, F, N>::is_valid
&&
is_sender<
typename bulk_execute_member<S, F, N>::result_type
>::value
)
>::type> :
bulk_execute_member<S, F, N>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_member);
};

template <typename S, typename F, typename N>
struct call_traits<S, void(F, N),
typename enable_if<
(
is_convertible<N, std::size_t>::value
&&
!bulk_execute_member<S, F, N>::is_valid
&&
bulk_execute_free<S, F, N>::is_valid
&&
is_sender<
typename bulk_execute_free<S, F, N>::result_type
>::value
)
>::type> :
bulk_execute_free<S, F, N>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

template <typename S, typename F, typename N>
struct call_traits<S, void(F, N),
typename enable_if<
(
is_convertible<N, std::size_t>::value
&&
!bulk_execute_member<S, F, N>::is_valid
&&
!bulk_execute_free<S, F, N>::is_valid
&&
is_sender<S>::value
&&
is_same<
typename result_of<
F(typename executor_index<typename remove_cvref<S>::type>::type)
>::type,
typename result_of<
F(typename executor_index<typename remove_cvref<S>::type>::type)
>::type
>::value
&&
static_require<S, bulk_guarantee_t::unsequenced_t>::is_valid
)
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = adapter);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef bulk_sender<S, F, N> result_type;
};

struct impl
{
#if defined(BOOST_ASIO_HAS_MOVE)
template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(F, N)>::overload == call_member,
typename call_traits<S, void(F, N)>::result_type
>::type
operator()(S&& s, F&& f, N&& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(F, N)>::is_noexcept))
{
return BOOST_ASIO_MOVE_CAST(S)(s).bulk_execute(
BOOST_ASIO_MOVE_CAST(F)(f), BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(F, N)>::overload == call_free,
typename call_traits<S, void(F, N)>::result_type
>::type
operator()(S&& s, F&& f, N&& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(F, N)>::is_noexcept))
{
return bulk_execute(BOOST_ASIO_MOVE_CAST(S)(s),
BOOST_ASIO_MOVE_CAST(F)(f), BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(F, N)>::overload == adapter,
typename call_traits<S, void(F, N)>::result_type
>::type
operator()(S&& s, F&& f, N&& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(F, N)>::is_noexcept))
{
return typename call_traits<S, void(F, N)>::result_type(
BOOST_ASIO_MOVE_CAST(S)(s), BOOST_ASIO_MOVE_CAST(F)(f),
BOOST_ASIO_MOVE_CAST(N)(n));
}
#else 
template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(const F&, const N&)>::overload == call_member,
typename call_traits<S, void(const F&, const N&)>::result_type
>::type
operator()(S& s, const F& f, const N& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(const F&, const N&)>::is_noexcept))
{
return s.bulk_execute(BOOST_ASIO_MOVE_CAST(F)(f),
BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(const F&, const N&)>::overload == call_member,
typename call_traits<S, void(const F&, const N&)>::result_type
>::type
operator()(const S& s, const F& f, const N& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(const F&, const N&)>::is_noexcept))
{
return s.bulk_execute(BOOST_ASIO_MOVE_CAST(F)(f),
BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(const F&, const N&)>::overload == call_free,
typename call_traits<S, void(const F&, const N&)>::result_type
>::type
operator()(S& s, const F& f, const N& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(const F&, const N&)>::is_noexcept))
{
return bulk_execute(s, BOOST_ASIO_MOVE_CAST(F)(f),
BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(const F&, const N&)>::overload == call_free,
typename call_traits<S, void(const F&, const N&)>::result_type
>::type
operator()(const S& s, const F& f, const N& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(const F&, const N&)>::is_noexcept))
{
return bulk_execute(s, BOOST_ASIO_MOVE_CAST(F)(f),
BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(const F&, const N&)>::overload == adapter,
typename call_traits<S, void(const F&, const N&)>::result_type
>::type
operator()(S& s, const F& f, const N& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(const F&, const N&)>::is_noexcept))
{
return typename call_traits<S, void(const F&, const N&)>::result_type(
s, BOOST_ASIO_MOVE_CAST(F)(f), BOOST_ASIO_MOVE_CAST(N)(n));
}

template <typename S, typename F, typename N>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(const F&, const N&)>::overload == adapter,
typename call_traits<S, void(const F&, const N&)>::result_type
>::type
operator()(const S& s, const F& f, const N& n) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(const F&, const N&)>::is_noexcept))
{
return typename call_traits<S, void(const F&, const N&)>::result_type(
s, BOOST_ASIO_MOVE_CAST(F)(f), BOOST_ASIO_MOVE_CAST(N)(n));
}
#endif 
};

template <typename T = impl>
struct static_instance
{
static const T instance;
};

template <typename T>
const T static_instance<T>::instance = {};

} 
namespace boost {
namespace asio {
namespace execution {
namespace {

static BOOST_ASIO_CONSTEXPR
const asio_execution_bulk_execute_fn::impl& bulk_execute =
asio_execution_bulk_execute_fn::static_instance<>::instance;

} 

template <typename S, typename F, typename N>
struct can_bulk_execute :
integral_constant<bool,
asio_execution_bulk_execute_fn::call_traits<S, void(F, N)>::overload !=
asio_execution_bulk_execute_fn::ill_formed>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename F, typename N>
constexpr bool can_bulk_execute_v = can_bulk_execute<S, F, N>::value;

#endif 

template <typename S, typename F, typename N>
struct is_nothrow_bulk_execute :
integral_constant<bool,
asio_execution_bulk_execute_fn::call_traits<S, void(F, N)>::is_noexcept>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename F, typename N>
constexpr bool is_nothrow_bulk_execute_v
= is_nothrow_bulk_execute<S, F, N>::value;

#endif 

template <typename S, typename F, typename N>
struct bulk_execute_result
{
typedef typename asio_execution_bulk_execute_fn::call_traits<
S, void(F, N)>::result_type type;
};

} 
} 
} 

#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
