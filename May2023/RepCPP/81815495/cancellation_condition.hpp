
#ifndef ASIO_EXPERIMENTAL_CANCELLATION_CONDITION_HPP
#define ASIO_EXPERIMENTAL_CANCELLATION_CONDITION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <exception>
#include "asio/cancellation_type.hpp"
#include "asio/error_code.hpp"
#include "asio/detail/type_traits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace experimental {

class wait_for_all
{
public:
template <typename... Args>
ASIO_CONSTEXPR cancellation_type_t operator()(
Args&&...) const ASIO_NOEXCEPT
{
return cancellation_type::none;
}
};

class wait_for_one
{
public:
ASIO_CONSTEXPR explicit wait_for_one(
cancellation_type_t cancel_type = cancellation_type::all)
: cancel_type_(cancel_type)
{
}

template <typename... Args>
ASIO_CONSTEXPR cancellation_type_t operator()(
Args&&...) const ASIO_NOEXCEPT
{
return cancel_type_;
}

private:
cancellation_type_t cancel_type_;
};


class wait_for_one_success
{
public:
ASIO_CONSTEXPR explicit wait_for_one_success(
cancellation_type_t cancel_type = cancellation_type::all)
: cancel_type_(cancel_type)
{
}

ASIO_CONSTEXPR cancellation_type_t
operator()() const ASIO_NOEXCEPT
{
return cancel_type_;
}

template <typename E, typename... Args>
ASIO_CONSTEXPR typename constraint<
!is_same<typename decay<E>::type, asio::error_code>::value
&& !is_same<typename decay<E>::type, std::exception_ptr>::value,
cancellation_type_t
>::type operator()(const E&, Args&&...) const ASIO_NOEXCEPT
{
return cancel_type_;
}

template <typename E, typename... Args>
ASIO_CONSTEXPR typename constraint<
is_same<typename decay<E>::type, asio::error_code>::value
|| is_same<typename decay<E>::type, std::exception_ptr>::value,
cancellation_type_t
>::type operator()(const E& e, Args&&...) const ASIO_NOEXCEPT
{
return !!e ? cancellation_type::none : cancel_type_;
}

private:
cancellation_type_t cancel_type_;
};


class wait_for_one_error
{
public:
ASIO_CONSTEXPR explicit wait_for_one_error(
cancellation_type_t cancel_type = cancellation_type::all)
: cancel_type_(cancel_type)
{
}

ASIO_CONSTEXPR cancellation_type_t
operator()() const ASIO_NOEXCEPT
{
return cancellation_type::none;
}

template <typename E, typename... Args>
ASIO_CONSTEXPR typename constraint<
!is_same<typename decay<E>::type, asio::error_code>::value
&& !is_same<typename decay<E>::type, std::exception_ptr>::value,
cancellation_type_t
>::type operator()(const E&, Args&&...) const ASIO_NOEXCEPT
{
return cancellation_type::none;
}

template <typename E, typename... Args>
ASIO_CONSTEXPR typename constraint<
is_same<typename decay<E>::type, asio::error_code>::value
|| is_same<typename decay<E>::type, std::exception_ptr>::value,
cancellation_type_t
>::type operator()(const E& e, Args&&...) const ASIO_NOEXCEPT
{
return !!e ? cancel_type_ : cancellation_type::none;
}

private:
cancellation_type_t cancel_type_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
