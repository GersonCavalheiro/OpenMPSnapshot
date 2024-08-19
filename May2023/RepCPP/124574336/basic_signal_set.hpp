
#ifndef BOOST_ASIO_BASIC_SIGNAL_SET_HPP
#define BOOST_ASIO_BASIC_SIGNAL_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/any_io_executor.hpp>
#include <boost/asio/async_result.hpp>
#include <boost/asio/detail/handler_type_requirements.hpp>
#include <boost/asio/detail/io_object_impl.hpp>
#include <boost/asio/detail/non_const_lvalue.hpp>
#include <boost/asio/detail/signal_set_service.hpp>
#include <boost/asio/detail/throw_error.hpp>
#include <boost/asio/detail/type_traits.hpp>
#include <boost/asio/error.hpp>
#include <boost/asio/execution_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {


template <typename Executor = any_io_executor>
class basic_signal_set
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_signal_set<Executor1> other;
};


explicit basic_signal_set(const executor_type& ex)
: impl_(ex)
{
}


template <typename ExecutionContext>
explicit basic_signal_set(ExecutionContext& context,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
}


basic_signal_set(const executor_type& ex, int signal_number_1)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
boost::asio::detail::throw_error(ec, "add");
}


template <typename ExecutionContext>
basic_signal_set(ExecutionContext& context, int signal_number_1,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
boost::asio::detail::throw_error(ec, "add");
}


basic_signal_set(const executor_type& ex, int signal_number_1,
int signal_number_2)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
boost::asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
boost::asio::detail::throw_error(ec, "add");
}


template <typename ExecutionContext>
basic_signal_set(ExecutionContext& context, int signal_number_1,
int signal_number_2,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
boost::asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
boost::asio::detail::throw_error(ec, "add");
}


basic_signal_set(const executor_type& ex, int signal_number_1,
int signal_number_2, int signal_number_3)
: impl_(ex)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
boost::asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
boost::asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_3, ec);
boost::asio::detail::throw_error(ec, "add");
}


template <typename ExecutionContext>
basic_signal_set(ExecutionContext& context, int signal_number_1,
int signal_number_2, int signal_number_3,
typename enable_if<
is_convertible<ExecutionContext&, execution_context&>::value
>::type* = 0)
: impl_(context)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
boost::asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
boost::asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_3, ec);
boost::asio::detail::throw_error(ec, "add");
}


~basic_signal_set()
{
}

executor_type get_executor() BOOST_ASIO_NOEXCEPT
{
return impl_.get_executor();
}


void add(int signal_number)
{
boost::system::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number, ec);
boost::asio::detail::throw_error(ec, "add");
}


BOOST_ASIO_SYNC_OP_VOID add(int signal_number,
boost::system::error_code& ec)
{
impl_.get_service().add(impl_.get_implementation(), signal_number, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void remove(int signal_number)
{
boost::system::error_code ec;
impl_.get_service().remove(impl_.get_implementation(), signal_number, ec);
boost::asio::detail::throw_error(ec, "remove");
}


BOOST_ASIO_SYNC_OP_VOID remove(int signal_number,
boost::system::error_code& ec)
{
impl_.get_service().remove(impl_.get_implementation(), signal_number, ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void clear()
{
boost::system::error_code ec;
impl_.get_service().clear(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "clear");
}


BOOST_ASIO_SYNC_OP_VOID clear(boost::system::error_code& ec)
{
impl_.get_service().clear(impl_.get_implementation(), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


void cancel()
{
boost::system::error_code ec;
impl_.get_service().cancel(impl_.get_implementation(), ec);
boost::asio::detail::throw_error(ec, "cancel");
}


BOOST_ASIO_SYNC_OP_VOID cancel(boost::system::error_code& ec)
{
impl_.get_service().cancel(impl_.get_implementation(), ec);
BOOST_ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
BOOST_ASIO_COMPLETION_TOKEN_FOR(void (boost::system::error_code, int))
SignalHandler BOOST_ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
BOOST_ASIO_INITFN_AUTO_RESULT_TYPE(SignalHandler,
void (boost::system::error_code, int))
async_wait(
BOOST_ASIO_MOVE_ARG(SignalHandler) handler
BOOST_ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<SignalHandler, void (boost::system::error_code, int)>(
initiate_async_wait(this), handler);
}

private:
basic_signal_set(const basic_signal_set&) BOOST_ASIO_DELETED;
basic_signal_set& operator=(const basic_signal_set&) BOOST_ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_signal_set* self)
: self_(self)
{
}

executor_type get_executor() const BOOST_ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename SignalHandler>
void operator()(BOOST_ASIO_MOVE_ARG(SignalHandler) handler) const
{
BOOST_ASIO_SIGNAL_HANDLER_CHECK(SignalHandler, handler) type_check;

detail::non_const_lvalue<SignalHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(),
handler2.value, self_->impl_.get_executor());
}

private:
basic_signal_set* self_;
};

detail::io_object_impl<detail::signal_set_service, Executor> impl_;
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
