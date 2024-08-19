
#ifndef ASIO_POSIX_BASIC_DESCRIPTOR_HPP
#define ASIO_POSIX_BASIC_DESCRIPTOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_POSIX_STREAM_DESCRIPTOR) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/handler_type_requirements.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/non_const_lvalue.hpp"
#include "asio/detail/reactive_descriptor_service.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"
#include "asio/posix/descriptor_base.hpp"

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace posix {


template <typename Executor = any_io_executor>
class basic_descriptor
: public descriptor_base
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_descriptor<Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#else
typedef detail::reactive_descriptor_service::native_handle_type
native_handle_type;
#endif

typedef basic_descriptor lowest_layer_type;


explicit basic_descriptor(const executor_type& ex)
: impl_(0, ex)
{
}


template <typename ExecutionContext>
explicit basic_descriptor(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
}


basic_descriptor(const executor_type& ex,
const native_handle_type& native_descriptor)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_descriptor, ec);
asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_descriptor(ExecutionContext& context,
const native_handle_type& native_descriptor,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(0, 0, context)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_descriptor, ec);
asio::detail::throw_error(ec, "assign");
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)

basic_descriptor(basic_descriptor&& other) ASIO_NOEXCEPT
: impl_(std::move(other.impl_))
{
}


basic_descriptor& operator=(basic_descriptor&& other)
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


void assign(const native_handle_type& native_descriptor)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(),
native_descriptor, ec);
asio::detail::throw_error(ec, "assign");
}


ASIO_SYNC_OP_VOID assign(const native_handle_type& native_descriptor,
asio::error_code& ec)
{
impl_.get_service().assign(
impl_.get_implementation(), native_descriptor, ec);
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


native_handle_type release()
{
return impl_.get_service().release(impl_.get_implementation());
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


template <typename IoControlCommand>
void io_control(IoControlCommand& command)
{
asio::error_code ec;
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
asio::detail::throw_error(ec, "io_control");
}


template <typename IoControlCommand>
ASIO_SYNC_OP_VOID io_control(IoControlCommand& command,
asio::error_code& ec)
{
impl_.get_service().io_control(impl_.get_implementation(), command, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool non_blocking() const
{
return impl_.get_service().non_blocking(impl_.get_implementation());
}


void non_blocking(bool mode)
{
asio::error_code ec;
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
asio::detail::throw_error(ec, "non_blocking");
}


ASIO_SYNC_OP_VOID non_blocking(
bool mode, asio::error_code& ec)
{
impl_.get_service().non_blocking(impl_.get_implementation(), mode, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


bool native_non_blocking() const
{
return impl_.get_service().native_non_blocking(
impl_.get_implementation());
}


void native_non_blocking(bool mode)
{
asio::error_code ec;
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
asio::detail::throw_error(ec, "native_non_blocking");
}


ASIO_SYNC_OP_VOID native_non_blocking(
bool mode, asio::error_code& ec)
{
impl_.get_service().native_non_blocking(
impl_.get_implementation(), mode, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


void wait(wait_type w)
{
asio::error_code ec;
impl_.get_service().wait(impl_.get_implementation(), w, ec);
asio::detail::throw_error(ec, "wait");
}


ASIO_SYNC_OP_VOID wait(wait_type w, asio::error_code& ec)
{
impl_.get_service().wait(impl_.get_implementation(), w, ec);
ASIO_SYNC_OP_VOID_RETURN(ec);
}


template <
ASIO_COMPLETION_TOKEN_FOR(void (asio::error_code))
WaitHandler ASIO_DEFAULT_COMPLETION_TOKEN_TYPE(executor_type)>
ASIO_INITFN_AUTO_RESULT_TYPE(WaitHandler,
void (asio::error_code))
async_wait(wait_type w,
ASIO_MOVE_ARG(WaitHandler) handler
ASIO_DEFAULT_COMPLETION_TOKEN(executor_type))
{
return async_initiate<WaitHandler, void (asio::error_code)>(
initiate_async_wait(this), handler, w);
}

protected:

~basic_descriptor()
{
}

detail::io_object_impl<detail::reactive_descriptor_service, Executor> impl_;

private:
basic_descriptor(const basic_descriptor&) ASIO_DELETED;
basic_descriptor& operator=(const basic_descriptor&) ASIO_DELETED;

class initiate_async_wait
{
public:
typedef Executor executor_type;

explicit initiate_async_wait(basic_descriptor* self)
: self_(self)
{
}

executor_type get_executor() const ASIO_NOEXCEPT
{
return self_->get_executor();
}

template <typename WaitHandler>
void operator()(ASIO_MOVE_ARG(WaitHandler) handler, wait_type w) const
{
ASIO_WAIT_HANDLER_CHECK(WaitHandler, handler) type_check;

detail::non_const_lvalue<WaitHandler> handler2(handler);
self_->impl_.get_service().async_wait(
self_->impl_.get_implementation(), w,
handler2.value, self_->impl_.get_executor());
}

private:
basic_descriptor* self_;
};
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
