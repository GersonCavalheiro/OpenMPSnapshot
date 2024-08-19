
#ifndef ASIO_DETAIL_WIN_IOCP_OPERATION_HPP
#define ASIO_DETAIL_WIN_IOCP_OPERATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_IOCP)

#include "asio/detail/handler_tracking.hpp"
#include "asio/detail/op_queue.hpp"
#include "asio/detail/socket_types.hpp"
#include "asio/error_code.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class win_iocp_io_context;

class win_iocp_operation
: public OVERLAPPED
ASIO_ALSO_INHERIT_TRACKED_HANDLER
{
public:
typedef win_iocp_operation operation_type;

void complete(void* owner, const asio::error_code& ec,
std::size_t bytes_transferred)
{
func_(owner, this, ec, bytes_transferred);
}

void destroy()
{
func_(0, this, asio::error_code(), 0);
}

void reset()
{
Internal = 0;
InternalHigh = 0;
Offset = 0;
OffsetHigh = 0;
hEvent = 0;
ready_ = 0;
}

protected:
typedef void (*func_type)(
void*, win_iocp_operation*,
const asio::error_code&, std::size_t);

win_iocp_operation(func_type func)
: next_(0),
func_(func)
{
reset();
}

~win_iocp_operation()
{
}

private:
friend class op_queue_access;
friend class win_iocp_io_context;
win_iocp_operation* next_;
func_type func_;
long ready_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
