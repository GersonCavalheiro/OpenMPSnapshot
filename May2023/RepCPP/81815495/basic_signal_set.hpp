
#ifndef ASIO_BASIC_SIGNAL_SET_HPP
#define ASIO_BASIC_SIGNAL_SET_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/non_const_lvalue.hpp"
#include "asio/detail/signal_set_service.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"

#include "asio/detail/push_options.hpp"

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
: impl_(0, ex)
{
}


template <typename ExecutionContext>
explicit basic_signal_set(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
}


basic_signal_set(const executor_type& ex, int signal_number_1)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
asio::detail::throw_error(ec, "add");
}


template <typename ExecutionContext>
basic_signal_set(ExecutionContext& context, int signal_number_1,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
asio::detail::throw_error(ec, "add");
}


basic_signal_set(const executor_type& ex, int signal_number_1,
int signal_number_2)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
asio::detail::throw_error(ec, "add");
}


template <typename ExecutionContext>
basic_signal_set(ExecutionContext& context, int signal_number_1,
int signal_number_2,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
asio::detail::throw_error(ec, "add");
}


basic_signal_set(const executor_type& ex, int signal_number_1,
int signal_number_2, int signal_number_3)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_3, ec);
asio::detail::throw_error(ec, "add");
}


template <typename ExecutionContext>
basic_signal_set(ExecutionContext& context, int signal_number_1,
int signal_number_2, int signal_number_3,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number_1, ec);
asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_2, ec);
asio::detail::throw_error(ec, "add");
impl_.get_service().add(impl_.get_implementation(), signal_number_3, ec);
asio::detail::throw_error(ec, "add");
}


~basic_signal_set()
{
}

executor_type get_executor() ASIO_NOEXCEPT
{
return impl_.get_executor();
}


void add(int signal_number)
{
asio::error_code ec;
impl_.get_service().add(impl_.get_implementation(), signal_number, ec);
asio::detail::throw_error(ec, "add");
}


ASIO_SYNC_OP_VOID add(int signal_number,
asio::error_code& ec)
{
impl_.get_service().add(impl_.get_implementation(), signal_number, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void remove(int signal_number)
{
asio::error_code ec;
impl_.get_service().remove(impl_.get_implementation(), signal_number, ec);
asio::detail::throw_error(ec, "remove");
}


ASIO_SYNC_OP_VOID remove(int signal_number,
asio::error_code& ec)
{
impl_.get_service().remove(impl_.get_implementation(), signal_number, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void clear()
{
asio::error_code ec;
impl_.get_service().clear(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "clear");
}


ASIO_SYNC_OP_VOID clear(asio::error_code& ec)
{
impl_.get_service().clear(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void cancel()
{
asio::error_code ec;
impl_.get_service().cancel(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "cancel");
}


ASIO_SYNC_OP_VOID cancel(asio::error_code& ec)
{
impl_.get_service().cancel(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code, int))
SignalHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(SignalHandler,
void (asio::error_code, int))
async_wait(
ASIO_MOVE_ARG(SignalHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<SignalHandler, void (asio::error_code, int)>(
initiate_async_wait(this), handler);
}

private:
basic_signal_set(const basic_signal_set&) ASIO_DELETED;
basic_signal_set& operator=(const basic_signal_set&) ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_signal_set* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename SignalHandler>
void operator()(ASIO_MOVE_ARG(SignalHandler) handler) const
{
ASIO_SIGNAL_HANDLER_CHECK(SignalHandler, handler) type_check;

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

#include "asio/detail/pop_options.hpp"

#endif 
