
#ifndef BOOST_ASIO_DETAIL_SOCKET_HOLDER_HPP
#define BOOST_ASIO_DETAIL_SOCKET_HOLDER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/socket_ops.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class socket_holder
: private noncopyable
{
public:
socket_holder()
: socket_(invalid_socket)
{
}

explicit socket_holder(socket_type s)
: socket_(s)
{
}

~socket_holder()
{
if (socket_ != invalid_socket)
{
boost::system::error_code ec;
socket_ops::state_type state = 0;
socket_ops::close(socket_, state, true, ec);
}
}

socket_type get() const
{
return socket_;
}

void reset()
{
if (socket_ != invalid_socket)
{
boost::system::error_code ec;
socket_ops::state_type state = 0;
socket_ops::close(socket_, state, true, ec);
socket_ = invalid_socket;
}
}

void reset(socket_type s)
{
reset();
socket_ = s;
}

socket_type release()
{
socket_type tmp = socket_;
socket_ = invalid_socket;
return tmp;
}

private:
socket_type socket_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
