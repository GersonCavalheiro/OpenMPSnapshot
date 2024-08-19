
#ifndef ASIO_WINDOWS_BASIC_OVERLAPPED_HANDLE_HPP
#define ASIO_WINDOWS_BASIC_OVERLAPPED_HANDLE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_RANDOM_ACCESS_HANDLE) \
|| defined(ASIO_HAS_WINDOWS_STREAM_HANDLE) \
|| defined(GENERATING_DOCUMENTATION)

#include <cstddef>
#include "asio/any_io_executor.hpp"
#include "asio/async_result.hpp"
#include "asio/detail/io_object_impl.hpp"
#include "asio/detail/throw_error.hpp"
#include "asio/detail/win_iocp_handle_service.hpp"
#include "asio/error.hpp"
#include "asio/execution_context.hpp"

#if defined(ASIO_HAS_MOVE)
# include <utility>
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {


template <typename Executor = any_io_executor>
class basic_overlapped_handle
{
public:
typedef Executor executor_type;

template <typename Executor1>
struct rebind_executor
{
typedef basic_overlapped_handle<Executor1> other;
};

#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined native_handle_type;
#else
typedef asio::detail::win_iocp_handle_service::native_handle_type
native_handle_type;
#endif

typedef basic_overlapped_handle lowest_layer_type;


explicit basic_overlapped_handle(const executor_type& ex)
: impl_(0, ex)
{
}


template <typename ExecutionContext>
explicit basic_overlapped_handle(ExecutionContext& context,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value,
defaulted_constraint
>::type = defaulted_constraint())
: impl_(0, 0, context)
{
}


basic_overlapped_handle(const executor_type& ex,
const native_handle_type& native_handle)
: impl_(0, ex)
{
asio::error_code ec;
impl_.get_service().assign(impl_.get_implementation(), native_handle, ec);
asio::detail::throw_error(ec, "assign");
}


template <typename ExecutionContext>
basic_overlapped_handle(ExecutionContext& context,
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

basic_overlapped_handle(basic_overlapped_handle&& other)
: impl_(std::move(other.impl_))
{
}


basic_overlapped_handle& operator=(basic_overlapped_handle&& other)
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

protected:

~basic_overlapped_handle()
{
}

asio::detail::io_object_impl<
asio::detail::win_iocp_handle_service, Executor> impl_;

private:
basic_overlapped_handle(const basic_overlapped_handle&) ASIO_DELETED;
basic_overlapped_handle& operator=(
const basic_overlapped_handle&) ASIO_DELETED;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
