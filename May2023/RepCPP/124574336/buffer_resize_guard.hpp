
#ifndef BOOST_ASIO_DETAIL_BUFFER_RESIZE_GUARD_HPP
#define BOOST_ASIO_DETAIL_BUFFER_RESIZE_GUARD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/limits.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename Buffer>
class buffer_resize_guard
{
public:
buffer_resize_guard(Buffer& buffer)
: buffer_(buffer),
old_size_(buffer.size())
{
}

~buffer_resize_guard()
{
if (old_size_ != (std::numeric_limits<size_t>::max)())
{
buffer_.resize(old_size_);
}
}

void commit()
{
old_size_ = (std::numeric_limits<size_t>::max)();
}

private:
Buffer& buffer_;

size_t old_size_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
