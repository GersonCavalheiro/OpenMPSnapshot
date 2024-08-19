
#ifndef ASIO_DETAIL_POSIX_FD_SET_ADAPTER_HPP
#define ASIO_DETAIL_POSIX_FD_SET_ADAPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS) \
&& !defined(__CYGWIN__) \
&& !defined(ASIO_WINDOWS_RUNTIME)

#include <cstring>
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/reactor_op_queue.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class posix_fd_set_adapter : noncopyable
{
public:
posix_fd_set_adapter()
: max_descriptor_(invalid_socket)
{
using namespace std; 
FD_ZERO(&fd_set_);
}

void reset()
{
using namespace std; 
FD_ZERO(&fd_set_);
}

bool set(socket_type descriptor)
{
if (descriptor < (socket_type)FD_SETSIZE)
{
if (max_descriptor_ == invalid_socket || descriptor > max_descriptor_)
max_descriptor_ = descriptor;
FD_SET(descriptor, &fd_set_);
return true;
}
return false;
}

void set(reactor_op_queue<socket_type>& operations, op_queue<operation>& ops)
{
reactor_op_queue<socket_type>::iterator i = operations.begin();
while (i != operations.end())
{
reactor_op_queue<socket_type>::iterator op_iter = i++;
if (!set(op_iter->first))
{
asio::error_code ec(error::fd_set_failure);
operations.cancel_operations(op_iter, ops, ec);
}
}
}

bool is_set(socket_type descriptor) const
{
return FD_ISSET(descriptor, &fd_set_) != 0;
}

operator fd_set*()
{
return &fd_set_;
}

socket_type max_descriptor() const
{
return max_descriptor_;
}

void perform(reactor_op_queue<socket_type>& operations,
op_queue<operation>& ops) const
{
reactor_op_queue<socket_type>::iterator i = operations.begin();
while (i != operations.end())
{
reactor_op_queue<socket_type>::iterator op_iter = i++;
if (is_set(op_iter->first))
operations.perform_operations(op_iter, ops);
}
}

private:
mutable fd_set fd_set_;
socket_type max_descriptor_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
