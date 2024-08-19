
#ifndef BOOST_ASIO_EXECUTION_DETAIL_AS_RECEIVER_HPP
#define BOOST_ASIO_EXECUTION_DETAIL_AS_RECEIVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/traits/set_done_member.hpp>
#include <boost/asio/traits/set_error_member.hpp>
#include <boost/asio/traits/set_value_member.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace execution {
namespace detail {

template <typename Function, typename>
struct as_receiver
{
Function f_;

template <typename F>
explicit as_receiver(BOOST_ASIO_MOVE_ARG(F) f, int)
: f_(BOOST_ASIO_MOVE_CAST(F)(f))
{
}

#if defined(BOOST_ASIO_MSVC) && defined(BOOST_ASIO_HAS_MOVE)
as_receiver(as_receiver&& other)
: f_(BOOST_ASIO_MOVE_CAST(Function)(other.f_))
{
}
#endif 

void set_value()
BOOST_ASIO_NOEXCEPT_IF(noexcept(declval<Function&>()()))
{
f_();
}

template <typename E>
void set_error(E) BOOST_ASIO_NOEXCEPT
{
std::terminate();
}

void set_done() BOOST_ASIO_NOEXCEPT
{
}
};

template <typename T>
struct is_as_receiver : false_type
{
};

template <typename Function, typename T>
struct is_as_receiver<as_receiver<Function, T> > : true_type
{
};

} 
} 
namespace traits {

#if !defined(BOOST_ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <typename Function, typename T>
struct set_value_member<
boost::asio::execution::detail::as_receiver<Function, T>, void()>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
#if defined(BOOST_ASIO_HAS_NOEXCEPT)
BOOST_ASIO_STATIC_CONSTEXPR(bool,
is_noexcept = noexcept(declval<Function&>()()));
#else 
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
#endif 
typedef void result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

template <typename Function, typename T, typename E>
struct set_error_member<
boost::asio::execution::detail::as_receiver<Function, T>, E>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
typedef void result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)

template <typename Function, typename T>
struct set_done_member<
boost::asio::execution::detail::as_receiver<Function, T> >
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
typedef void result_type;
};

#endif 

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
