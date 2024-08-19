
#ifndef BOOST_ASIO_BASIC_DEADLINE_TIMER_HPP
#define BOOST_ASIO_BASIC_DEADLINE_TIMER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_BOOST_DATE_TIME) \
|| defined(GENERATING_DOCUMENTATION)

#include <cstddef>
#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/detail/deadline_timer_service.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/io_object_impl.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>
#include <boost/asio/time_traits.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {


template <typename Time,
typename TimeTraits = boost::asio::time_traits<Time>,
typename Executor = any_io_executor>
class basic_deadline_timer
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_deadline_timer<Time, TimeTraits, Executor1> other;
};

typedef TimeTraits traits_type;

typedef typename traits_type::time_type time_type;

typedef typename traits_type::duration_type duration_type;


explicit basic_deadline_timer(const executor_type& ex)
: impl_(ex)
{
}


template <typename ExecutionContext>
explicit basic_deadline_timer(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
}


basic_deadline_timer(const executor_type& ex, const time_type& expiry_time)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().expires_at(impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_at");
}


template <typename ExecutionContext>
basic_deadline_timer(ExecutionContext& context, const time_type& expiry_time,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().expires_at(impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_at");
}


basic_deadline_timer(const executor_type& ex,
const duration_type& expiry_time)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().expires_from_now(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_from_now");
}


template <typename ExecutionContext>
basic_deadline_timer(ExecutionContext& context,
const duration_type& expiry_time,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().expires_from_now(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_from_now");
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_deadline_timer(basic_deadline_timer&& other)
: impl_(std::move(other.impl_))
{
}


basic_deadline_timer& operator=(basic_deadline_timer&& other)
{
impl_ = std::move(other.impl_);
return *this;
}
#endif 


~basic_deadline_timer()
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


std::size_t cancel(boost::system::error_code& ec)
{
return impl_.get_service().cancel(impl_.get_implementation(), ec);
}


std::size_t cancel_one()
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().cancel_one(
impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "cancel_one");
return s;
}


std::size_t cancel_one(boost::system::error_code& ec)
{
return impl_.get_service().cancel_one(impl_.get_implementation(), ec);
}


time_type expires_at() const
{
return impl_.get_service().expires_at(impl_.get_implementation());
}


std::size_t expires_at(const time_type& expiry_time)
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().expires_at(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_at");
return s;
}


std::size_t expires_at(const time_type& expiry_time,
boost::system::error_code& ec)
{
return impl_.get_service().expires_at(
impl_.get_implementation(), expiry_time, ec);
}


duration_type expires_from_now() const
{
return impl_.get_service().expires_from_now(impl_.get_implementation());
}


std::size_t expires_from_now(const duration_type& expiry_time)
{
boost::system::error_code ec;
std::size_t s = impl_.get_service().expires_from_now(
impl_.get_implementation(), expiry_time, ec);
boost::asio::detail::throw_error(ec, "expires_from_now");
return s;
}


std::size_t expires_from_now(const duration_type& expiry_time,
boost::system::error_code& ec)
{
return impl_.get_service().expires_from_now(
impl_.get_implementation(), expiry_time, ec);
}


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
basic_deadline_timer(const basic_deadline_timer&) BOOST_ASIO_DELETED;
basic_deadline_timer& operator=(
const basic_deadline_timer&) BOOST_ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_deadline_timer* self)
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
basic_deadline_timer* self_;
};

detail::io_object_impl<
detail::deadline_timer_service<TimeTraits>, Executor> impl_;
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
