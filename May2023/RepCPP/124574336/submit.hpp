
#ifndef BOOST_ASIO_EXECUTION_SUBMIT_HPP
#define BOOST_ASIO_EXECUTION_SUBMIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/execution/detail/submit_receiver.hpp>
#include <boost/asio/execution/executor.hpp>
#include <boost/asio/execution/receiver.hpp>
#include <boost/asio/execution/sender.hpp>
#include <boost/asio/execution/start.hpp>
#include <boost/asio/traits/submit_member.hpp>
#include <boost/asio/traits/submit_free.hpp>

#include <boost/asio/detail/push_options.hpp>

#if defined(GENERATING_DOCUMENTATION)

namespace boost {
namespace asio {
namespace execution {


inline constexpr unspecified submit = unspecified;


template <typename S, typename R>
struct can_submit :
integral_constant<bool, automatically_determined>
{
};

} 
} 
} 

#else 

namespace asio_execution_submit_fn {

using boost::asio::declval;
using boost::asio::enable_if;
using boost::asio::execution::is_sender_to;
using boost::asio::traits::submit_free;
using boost::asio::traits::submit_member;

void submit();

enum overload_type
{
call_member,
call_free,
adapter,
ill_formed
};

template <typename S, typename R, typename = void>
struct call_traits
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = ill_formed);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

template <typename S, typename R>
struct call_traits<S, void(R),
typename enable_if<
(
submit_member<S, R>::is_valid
&&
is_sender_to<S, R>::value
)
>::type> :
submit_member<S, R>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_member);
};

template <typename S, typename R>
struct call_traits<S, void(R),
typename enable_if<
(
!submit_member<S, R>::is_valid
&&
submit_free<S, R>::is_valid
&&
is_sender_to<S, R>::value
)
>::type> :
submit_free<S, R>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = call_free);
};

template <typename S, typename R>
struct call_traits<S, void(R),
typename enable_if<
(
!submit_member<S, R>::is_valid
&&
!submit_free<S, R>::is_valid
&&
is_sender_to<S, R>::value
)
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(overload_type, overload = adapter);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
typedef void result_type;
};

struct impl
{
#if defined(BOOST_ASIO_HAS_MOVE)
template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(R)>::overload == call_member,
typename call_traits<S, void(R)>::result_type
>::type
operator()(S&& s, R&& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(R)>::is_noexcept))
{
return BOOST_ASIO_MOVE_CAST(S)(s).submit(BOOST_ASIO_MOVE_CAST(R)(r));
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(R)>::overload == call_free,
typename call_traits<S, void(R)>::result_type
>::type
operator()(S&& s, R&& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(R)>::is_noexcept))
{
return submit(BOOST_ASIO_MOVE_CAST(S)(s), BOOST_ASIO_MOVE_CAST(R)(r));
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S, void(R)>::overload == adapter,
typename call_traits<S, void(R)>::result_type
>::type
operator()(S&& s, R&& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S, void(R)>::is_noexcept))
{
return boost::asio::execution::start(
(new boost::asio::execution::detail::submit_receiver<S, R>(
BOOST_ASIO_MOVE_CAST(S)(s), BOOST_ASIO_MOVE_CAST(R)(r)))->state_);
}
#else 
template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S&, void(R&)>::overload == call_member,
typename call_traits<S&, void(R&)>::result_type
>::type
operator()(S& s, R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S&, void(R&)>::is_noexcept))
{
return s.submit(r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const S&, void(R&)>::overload == call_member,
typename call_traits<const S&, void(R&)>::result_type
>::type
operator()(const S& s, R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const S&, void(R&)>::is_noexcept))
{
return s.submit(r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S&, void(R&)>::overload == call_free,
typename call_traits<S&, void(R&)>::result_type
>::type
operator()(S& s, R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S&, void(R&)>::is_noexcept))
{
return submit(s, r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const S&, void(R&)>::overload == call_free,
typename call_traits<const S&, void(R&)>::result_type
>::type
operator()(const S& s, R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const S&, void(R&)>::is_noexcept))
{
return submit(s, r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S&, void(R&)>::overload == adapter,
typename call_traits<S&, void(R&)>::result_type
>::type
operator()(S& s, R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S&, void(R&)>::is_noexcept))
{
return boost::asio::execution::start(
(new boost::asio::execution::detail::submit_receiver<
S&, R&>(s, r))->state_);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const S&, void(R&)>::overload == adapter,
typename call_traits<const S&, void(R&)>::result_type
>::type
operator()(const S& s, R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const S&, void(R&)>::is_noexcept))
{
boost::asio::execution::start(
(new boost::asio::execution::detail::submit_receiver<
const S&, R&>(s, r))->state_);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S&, void(const R&)>::overload == call_member,
typename call_traits<S&, void(const R&)>::result_type
>::type
operator()(S& s, const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S&, void(const R&)>::is_noexcept))
{
return s.submit(r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const S&, void(const R&)>::overload == call_member,
typename call_traits<const S&, void(const R&)>::result_type
>::type
operator()(const S& s, const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const S&, void(const R&)>::is_noexcept))
{
return s.submit(r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S&, void(const R&)>::overload == call_free,
typename call_traits<S&, void(const R&)>::result_type
>::type
operator()(S& s, const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S&, void(const R&)>::is_noexcept))
{
return submit(s, r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const S&, void(const R&)>::overload == call_free,
typename call_traits<const S&, void(const R&)>::result_type
>::type
operator()(const S& s, const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const S&, void(const R&)>::is_noexcept))
{
return submit(s, r);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<S&, void(const R&)>::overload == adapter,
typename call_traits<S&, void(const R&)>::result_type
>::type
operator()(S& s, const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<S&, void(const R&)>::is_noexcept))
{
boost::asio::execution::start(
(new boost::asio::execution::detail::submit_receiver<
S&, const R&>(s, r))->state_);
}

template <typename S, typename R>
BOOST_ASIO_CONSTEXPR typename enable_if<
call_traits<const S&, void(const R&)>::overload == adapter,
typename call_traits<const S&, void(const R&)>::result_type
>::type
operator()(const S& s, const R& r) const
BOOST_ASIO_NOEXCEPT_IF((
call_traits<const S&, void(const R&)>::is_noexcept))
{
boost::asio::execution::start(
(new boost::asio::execution::detail::submit_receiver<
const S&, const R&>(s, r))->state_);
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

static BOOST_ASIO_CONSTEXPR const asio_execution_submit_fn::impl&
submit = asio_execution_submit_fn::static_instance<>::instance;

} 

template <typename S, typename R>
struct can_submit :
integral_constant<bool,
asio_execution_submit_fn::call_traits<S, void(R)>::overload !=
asio_execution_submit_fn::ill_formed>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
constexpr bool can_submit_v = can_submit<S, R>::value;

#endif 

template <typename S, typename R>
struct is_nothrow_submit :
integral_constant<bool,
asio_execution_submit_fn::call_traits<S, void(R)>::is_noexcept>
{
};

#if defined(BOOST_ASIO_HAS_VARIABLE_TEMPLATES)

template <typename S, typename R>
constexpr bool is_nothrow_submit_v
= is_nothrow_submit<S, R>::value;

#endif 

template <typename S, typename R>
struct submit_result
{
typedef typename asio_execution_submit_fn::call_traits<
S, void(R)>::result_type type;
};

namespace detail {

template <typename S, typename R>
void submit_helper(BOOST_ASIO_MOVE_ARG(S) s, BOOST_ASIO_MOVE_ARG(R) r)
{
execution::submit(BOOST_ASIO_MOVE_CAST(S)(s), BOOST_ASIO_MOVE_CAST(R)(r));
}

} 
} 
} 
} 

#endif 

#include <boost/asio/detail/pop_options.hpp>

#endif 
