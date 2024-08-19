
#ifndef ASIO_WINDOWS_OVERLAPPED_PTR_HPP
#define ASIO_WINDOWS_OVERLAPPED_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_WINDOWS_OVERLAPPED_PTR) \
|| defined(GENERATING_DOCUMENTATION)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/win_iocp_overlapped_ptr.hpp"
#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace windows {


class overlapped_ptr
: private noncopyable
{
public:
overlapped_ptr()
: impl_()
{
}

template <typename ExecutionContext, typename Handler>
explicit overlapped_ptr(ExecutionContext& context,
ASIO_MOVE_ARG(Handler) handler,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
: impl_(context.get_executor(), ASIO_MOVE_CAST(Handler)(handler))
{
}

template <typename Executor, typename Handler>
explicit overlapped_ptr(const Executor& ex,
ASIO_MOVE_ARG(Handler) handler,
typename constraint<
execution::is_executor<Executor>::value
|| is_executor<Executor>::value
>::type = 0)
: impl_(ex, ASIO_MOVE_CAST(Handler)(handler))
{
}

~overlapped_ptr()
{
}

void reset()
{
impl_.reset();
}

template <typename ExecutionContext, typename Handler>
void reset(ExecutionContext& context, ASIO_MOVE_ARG(Handler) handler,
typename constraint<
is_convertible<ExecutionContext&, execution_context&>::value
>::type = 0)
{
impl_.reset(context.get_executor(), ASIO_MOVE_CAST(Handler)(handler));
}

template <typename Executor, typename Handler>
void reset(const Executor& ex, ASIO_MOVE_ARG(Handler) handler,
typename constraint<
execution::is_executor<Executor>::value
|| is_executor<Executor>::value
>::type = 0)
{
impl_.reset(ex, ASIO_MOVE_CAST(Handler)(handler));
}

OVERLAPPED* get()
{
return impl_.get();
}

const OVERLAPPED* get() const
{
return impl_.get();
}

OVERLAPPED* release()
{
return impl_.release();
}

void complete(const asio::error_code& ec,
std::size_t bytes_transferred)
{
impl_.complete(ec, bytes_transferred);
}

private:
detail::win_iocp_overlapped_ptr impl_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
