
#ifndef BOOST_ASIO_EXECUTION_SET_DONE_HPP
#define BOOST_ASIO_EXECUTION_SET_DONE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/traits/set_done_member.hpp>
#include <boost/asio/traits/set_done_free.hpp>

#include <boost/asio/detail/push_options.hpp>

#if defined(GENERATING_DOCUMENTATION)

namespace boost {
namespace asio {
namespace execution {


inline constexpr unspecified set_done = unspecified;


template <typename R>
struct can_set_done :
integral_constant<bool, automatically_determined>
{
};

} 
} 
} 

#else 

namespace asio_execution_set_done_fn {

using boost::asio::decay;
using boost::asio::declval;
using boost::asio::enable_if;
using boost::asio::traits::set_done_free;
using boost::asio::traits::set_done_member;

void set_done();

enum overload_type
{
call_member,
call_free,
ill_formed
};

template <typename R, typename = void>
struct call_traits
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = ill_formed);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

template <typename R>
struct call_traits<R,
typename enable_if<
(
set_done_member<R>::is_valid
)
>::type> :
set_done_member<R>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_member);
};

template <typename R>
struct call_traits<R,
typename enable_if<
(
!set_done_member<R>::is_valid
&&
set_done_free<R>::is_valid
)
>::type> :
set_done_free<R>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

struct impl
{
#if defined(BOOST_ASIO_HAS_MOVE)
template <typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<R>::overload == call_member,
typename call_traits<R>::result_type
>::type
operator()(R&& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<R>::is_noexcept))
{
return BOOST_ASIO_MOVE_CAST(R)(r).set_done();
}

template <typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<R>::overload == call_free,
typename call_traits<R>::result_type
>::type
operator()(R&& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<R>::is_noexcept))
{
return set_done(BOOST_ASIO_MOVE_CAST(R)(r));
}
#else 
template <typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<R&>::overload == call_member,
typename call_traits<R&>::result_type
>::type
operator()(R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<R&>::is_noexcept))
{
return r.set_done();
}

template <typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const R&>::overload == call_member,
typename call_traits<const R&>::result_type
>::type
operator()(const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const R&>::is_noexcept))
{
return r.set_done();
}

template <typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<R&>::overload == call_free,
typename call_traits<R&>::result_type
>::type
operator()(R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<R&>::is_noexcept))
{
return set_done(r);
}

template <typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const R&>::overload == call_free,
typename call_traits<const R&>::result_type
>::type
operator()(const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const R&>::is_noexcept))
{
return set_done(r);
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

static BOOST_ASIO_CONSTEXPR const asio_execution_set_done_fn::impl&
set_done = asio_execution_set_done_fn::static_instance<>::instance;

} 

template <typename R>
struct can_set_done :
integral_constant<bool,
asio_execution_set_done_fn::call_traits<R>::overload !=
asio_execution_set_done_fn::ill_formed>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename R>
constexpr bool can_set_done_v = can_set_done<R>::value;

#endif 

template <typename R>
struct is_nothrow_set_done :
integral_constant<bool,
asio_execution_set_done_fn::call_traits<R>::is_noexcept>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename R>
constexpr bool is_nothrow_set_done_v
= is_nothrow_set_done<R>::value;

#endif 

} 
} 
} 

#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
