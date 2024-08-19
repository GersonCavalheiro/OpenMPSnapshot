
#ifndef BOOST_ASIO_BASIC_WAITABLE_TIMER_HPP
#define BOOST_ASIO_BASIC_WAITABLE_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/detail/chrono_time_traits.hpp>
#include <boost/asio/detail/deadline_timer_service.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/io_object_impl.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/wait_traits.hpp>

#if defined(BOOST_ASIO_HAS_MOVE)
# include <utility>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if !defined(BOOST_ASIO_BASIC_WAITABLE_TIMER_FWD_DECL)
#define BOOST_ASIO_BASIC_WAITABLE_TIMER_FWD_DECL

template <typename Clock,
typename WaitTraits = boost::asio::wait_traits<Clock>,
typename Executor = any_io_executor>
class basic_waitable_timer;

#endif 


template <typename Clock, typename WaitTraits, typename Executor>
class basic_waitable_timer
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_waitable_timer<Clock, WaitTraits, Executor1> other;
};

typedef Clock clock_type;

typedef typename clock_type::duration duration;

typedef typename clock_type::time_point time_point;

typedef WaitTraits traits_type;


explicit basic_waitable_timer(const executor_type& ex)
: impl_(ex)
{
}


template <typename ExecutionContext>
explicit basic_waitable_timer(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
}


basic_waitable_timer(const executor_type& ex, const time_point& expiry_time)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().expires_at(impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_at");
}


template <typename ExecutionContext>
explicit basic_waitable_timer(ExecutionContext& context,
const time_point& expiry_time,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().expires_at(impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_at");
}


basic_waitable_timer(const executor_type& ex, const duration& expiry_time)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().expires_after(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_after");
}


template <typename ExecutionContext>
explicit basic_waitable_timer(ExecutionContext& context,
const duration& expiry_time,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().expires_after(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_after");
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_waitable_timer(basic_waitable_timer&& other)
: impl_(std::move(other.impl_))
{
}


basic_waitable_timer& operator=(basic_waitable_timer&& other)
{
impl_ = std::move(other.impl_);
return *this;
}

template <typename Clock1, typename WaitTraits1, typename Executor1>
friend class basic_waitable_timer;


template <typename Executor1>
basic_waitable_timer(
basic_waitable_timer<Clock, WaitTraits, Executor1>&& other,
typename enable_if<
is_convertible<Executor1, Executor>::value
>::type* = 0)
: impl_(std::move(other.impl_))
{
}


template <typename Executor1>
typename enable_if<
is_convertible<Executor1, Executor>::value,
basic_waitable_timer&
>::type operator=(basic_waitable_timer<Clock, WaitTraits, Executor1>&& other)
{
basic_waitable_timer tmp(std::move(other));
impl_ = std::move(tmp.impl_);
return *this;
}
#endif 


~basic_waitable_timer()
{
}

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return impl_.get_executor();
}


std::size_t cancel()
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().cancel(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "cancel");
return s;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

std::size_t cancel(boost::system::error_code& ec)
{
return impl_.get_service().cancel(impl_.get_implementation(), ec);
}
#endif 


std::size_t cancel_one()
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().cancel_one(
impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "cancel_one");
return s;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

std::size_t cancel_one(boost::system::error_code& ec)
{
return impl_.get_service().cancel_one(impl_.get_implementation(), ec);
}


time_point expires_at() const
{
return impl_.get_service().expires_at(impl_.get_implementation());
}
#endif 


time_point expiry() const
{
return impl_.get_service().expiry(impl_.get_implementation());
}


std::size_t expires_at(const time_point& expiry_time)
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().expires_at(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_at");
return s;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

std::size_t expires_at(const time_point& expiry_time,
boost::system::error_code& ec)
{
return impl_.get_service().expires_at(
impl_.get_implementation(), expiry_time, ec);
}
#endif 


std::size_t expires_after(const duration& expiry_time)
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().expires_after(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_after");
return s;
}

#if !defined(BOOST_ASIO_NO_DEPRECATED)

duration expires_from_now() const
{
return impl_.get_service().expires_from_now(impl_.get_implementation());
}


std::size_t expires_from_now(const duration& expiry_time)
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().expires_from_now(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_from_now");
return s;
}


std::size_t expires_from_now(const duration& expiry_time,
boost::system::error_code& ec)
{
return impl_.get_service().expires_from_now(
impl_.get_implementation(), expiry_time, ec);
}
#endif 


void wait()
{
boost::system::error_code ec;
impl_.get_service().wait(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "wait");
}


void wait(boost::system::error_code& ec)
{
impl_.get_service().wait(impl_.get_implementation(), ec);
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code))
WaitHandler BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler,
void (boost::system::error_code))
async_wait(
BOOST_ASIO_MOVE_ARG(WaitHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WaitHandler, void (boost::system::error_code)>(
initiate_async_wait(this), handler);
}

private:
basic_waitable_timer(const basic_waitable_timer&) BOOST_ASIO_DELETED;
basic_waitable_timer& operator=(
const basic_waitable_timer&) BOOST_ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_waitable_timer* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WaitHandler>
void operator()(BOOST_ASIO_MOVE_ARG(WaitHandler) handler) const
{
BOOST_ASIO_WAIT_HANDLER_CHECK(WaitHandler, handler) type_check;

detail::non_const_lvalue<WaitHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(),
handler2.value, self_->impl_.get_executor());
}

private:
basic_waitable_timer* self_;
};

detail::io_object_impl<
detail::deadline_timer_service<
detail::chrono_time_traits<Clock, WaitTraits> >,
executor_type > impl_;
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
