
#ifndef ASIO_WINDOWS_BASIC_OBJECT_HANDLE_HPP
#define ASIO_WINDOWS_BASIC_OBJECT_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_OBJECT_HANDLE) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/win_object_handle_service.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {


template <typename Executor = any_io_executor>
class basic_object_handle
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_object_handle<Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#else
typedef asio::detail::win_object_handle_service::native_handle_type
native_handle_type;
#endif

typedef basic_object_handle lowest_layer_type;


explicit basic_object_handle(const executor_type& ex)
: impl_(0, ex)
{
}


template <typename ExecutionContext>
explicit basic_object_handle(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
}


basic_object_handle(const executor_type& ex,
const native_handle_type& native_handle)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(), native_handle, ec);
asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_object_handle(ExecutionContext& context,
const native_handle_type& native_handle,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(), native_handle, ec);
asio::detail::throw_error(ec, "assign");
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_object_handle(basic_object_handle&& other)
: impl_(std::move(other.impl_))
{
}


basic_object_handle& operator=(basic_object_handle&& other)
{
impl_ = std::move(other.impl_);
return *this;
}
#endif 

executor_type get_executor() ASIO_NOEXCEPT
{
return impl_.get_executor();
}


lowest_layer_type& lowest_layer()
{
return *this;
}


const lowest_layer_type& lowest_layer() const
{
return *this;
}


void assign(const native_handle_type& handle)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(), handle, ec);
asio::detail::throw_error(ec, "assign");
}


ASIO_SYNC_OP_VOID assign(const native_handle_type& handle,
asio::error_code& ec)
{
impl_.get_service().assign(impl_.get_implementation(), handle, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}

bool is_open() const
{
return impl_.get_service().is_open(impl_.get_implementation());
}


void close()
{
asio::error_code ec;
impl_.get_service().close(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "close");
}


ASIO_SYNC_OP_VOID close(asio::error_code& ec)
{
impl_.get_service().close(impl_.get_implementation(), ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


native_handle_type native_handle()
{
return impl_.get_service().native_handle(impl_.get_implementation());
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


void wait()
{
asio::error_code ec;
impl_.get_service().wait(impl_.get_implementation(), ec);
asio::detail::throw_error(ec, "wait");
}


void wait(asio::error_code& ec)
{
impl_.get_service().wait(impl_.get_implementation(), ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code))
WaitHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler,
void (asio::error_code))
async_wait(
ASIO_MOVE_ARG(WaitHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WaitHandler, void (asio::error_code)>(
initiate_async_wait(this), handler);
}

private:
basic_object_handle(const basic_object_handle&) ASIO_DELETED;
basic_object_handle& operator=(const basic_object_handle&) ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_object_handle* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WaitHandler>
void operator()(ASIO_MOVE_ARG(WaitHandler) handler) const
{
ASIO_WAIT_HANDLER_CHECK(WaitHandler, handler) type_check;

detail::non_const_lvalue<WaitHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(),
handler2.value, self_->impl_.get_executor());
}

private:
basic_object_handle* self_;
};

asio::detail::io_object_impl<
asio::detail::win_object_handle_service, Executor> impl_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
