
#ifndef BOOST_ASIO_EXECUTION_DETAIL_VOID_RECEIVER_HPP
#define BOOST_ASIO_EXECUTION_DETAIL_VOID_RECEIVER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/traits/set_done_member.hpp>
#include <boost/asio/traits/set_error_member.hpp>
#include <boost/asio/traits/set_value_member.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace execution {
namespace detail {

struct void_receiver
{
void set_value() BOOST_ASIO_NOEXCEPT
{
}

template <typename E>
void set_error(BOOST_ASIO_MOVE_ARG(E)) BOOST_ASIO_NOEXCEPT
{
}

void set_done() BOOST_ASIO_NOEXCEPT
{
}
};

} 
} 
namespace traits {

#if !defined(BOOST_ASIO_HAS_DEDUCED_SET_VALUE_MEMBER_TRAIT)

template <>
struct set_value_member<boost::asio::execution::detail::void_receiver, void()>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
typedef void result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_SET_ERROR_MEMBER_TRAIT)

template <typename E>
struct set_error_member<boost::asio::execution::detail::void_receiver, E>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = true);
typedef void result_type;
};

#endif 

#if !defined(BOOST_ASIO_HAS_DEDUCED_SET_DONE_MEMBER_TRAIT)

template <>
struct set_done_member<boost::asio::execution::detail::void_receiver>
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
