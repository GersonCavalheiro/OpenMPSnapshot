
#ifndef BOOST_ASIO_DETAIL_BUFFERED_STREAM_STORAGE_HPP
#define BOOST_ASIO_DETAIL_BUFFERED_STREAM_STORAGE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/detail/assert.hpp>
#include <cstddef>
#include <cstring>
#include <vector>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class buffered_stream_storage
{
public:
typedef unsigned char byte_type;

typedef std::size_t size_type;

explicit buffered_stream_storage(std::size_t buffer_capacity)
: begin_offset_(0),
end_offset_(0),
buffer_(buffer_capacity)
{
}

void clear()
{
begin_offset_ = 0;
end_offset_ = 0;
}

mutable_buffer data()
{
return boost::asio::buffer(buffer_) + begin_offset_;
}

const_buffer data() const
{
return boost::asio::buffer(buffer_) + begin_offset_;
}

bool empty() const
{
return begin_offset_ == end_offset_;
}

size_type size() const
{
return end_offset_ - begin_offset_;
}

void resize(size_type length)
{
BOOST_ASIO_ASSERT(length <= capacity());
if (begin_offset_ + length <= capacity())
{
end_offset_ = begin_offset_ + length;
}
else
{
using namespace std; 
memmove(&buffer_[0], &buffer_[0] + begin_offset_, size());
end_offset_ = length;
begin_offset_ = 0;
}
}

size_type capacity() const
{
return buffer_.size();
}

void consume(size_type count)
{
BOOST_ASIO_ASSERT(begin_offset_ + count <= end_offset_);
begin_offset_ += count;
if (empty())
clear();
}

private:
size_type begin_offset_;

size_type end_offset_;

std::vector<byte_type> buffer_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
