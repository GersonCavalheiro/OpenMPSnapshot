
#ifndef BOOST_ASIO_DETAIL_IO_CONTROL_HPP
#define BOOST_ASIO_DETAIL_IO_CONTROL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {
namespace io_control {

class bytes_readable
{
public:
bytes_readable()
: value_(0)
{
}

bytes_readable(std::size_t value)
: value_(static_cast<detail::ioctl_arg_type>(value))
{
}

int name() const
{
return static_cast<int>(BOOST_ASIO_OS_DEF(FIONREAD));
}

void set(std::size_t value)
{
value_ = static_cast<detail::ioctl_arg_type>(value);
}

std::size_t get() const
{
return static_cast<std::size_t>(value_);
}

detail::ioctl_arg_type* data()
{
return &value_;
}

const detail::ioctl_arg_type* data() const
{
return &value_;
}

private:
detail::ioctl_arg_type value_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
