
#ifndef ASIO_DETAIL_CONSUMING_BUFFERS_HPP
#define ASIO_DETAIL_CONSUMING_BUFFERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>
#include "asio/buffer.hpp"
#include "asio/detail/buffer_sequence_adapter.hpp"
#include "asio/detail/limits.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Buffers>
struct prepared_buffers_max
{
enum { value = buffer_sequence_adapter_base::max_buffers };
};

template <typename Elem, std::size_t N>
struct prepared_buffers_max<boost::array<Elem, N> >
{
enum { value = N };
};

#if defined(ASIO_HAS_STD_ARRAY)

template <typename Elem, std::size_t N>
struct prepared_buffers_max<std::array<Elem, N> >
{
enum { value = N };
};

#endif 

template <typename Buffer, std::size_t MaxBuffers>
struct prepared_buffers
{
typedef Buffer value_type;
typedef const Buffer* const_iterator;

enum { max_buffers = MaxBuffers < 16 ? MaxBuffers : 16 };

prepared_buffers() : count(0) {}
const_iterator begin() const { return elems; }
const_iterator end() const { return elems + count; }

Buffer elems[max_buffers];
std::size_t count;
};

template <typename Buffer, typename Buffers, typename Buffer_Iterator>
class consuming_buffers
{
public:
typedef prepared_buffers<Buffer, prepared_buffers_max<Buffers>::value>
prepared_buffers_type;

explicit consuming_buffers(const Buffers& buffers)
: buffers_(buffers),
total_consumed_(0),
next_elem_(0),
next_elem_offset_(0)
{
using asio::buffer_size;
total_size_ = buffer_size(buffers);
}

bool empty() const
{
return total_consumed_ >= total_size_;
}

prepared_buffers_type prepare(std::size_t max_size)
{
prepared_buffers_type result;

Buffer_Iterator next = asio::buffer_sequence_begin(buffers_);
Buffer_Iterator end = asio::buffer_sequence_end(buffers_);

std::advance(next, next_elem_);
std::size_t elem_offset = next_elem_offset_;
while (next != end && max_size > 0 && (result.count) < result.max_buffers)
{
Buffer next_buf = Buffer(*next) + elem_offset;
result.elems[result.count] = asio::buffer(next_buf, max_size);
max_size -= result.elems[result.count].size();
elem_offset = 0;
if (result.elems[result.count].size() > 0)
++result.count;
++next;
}

return result;
}

void consume(std::size_t size)
{
total_consumed_ += size;

Buffer_Iterator next = asio::buffer_sequence_begin(buffers_);
Buffer_Iterator end = asio::buffer_sequence_end(buffers_);

std::advance(next, next_elem_);
while (next != end && size > 0)
{
Buffer next_buf = Buffer(*next) + next_elem_offset_;
if (size < next_buf.size())
{
next_elem_offset_ += size;
size = 0;
}
else
{
size -= next_buf.size();
next_elem_offset_ = 0;
++next_elem_;
++next;
}
}
}

std::size_t total_consumed() const
{
return total_consumed_;
}

private:
Buffers buffers_;
std::size_t total_size_;
std::size_t total_consumed_;
std::size_t next_elem_;
std::size_t next_elem_offset_;
};

template <typename Buffer>
class consuming_single_buffer
{
public:
template <typename Buffer1>
explicit consuming_single_buffer(const Buffer1& buffer)
: buffer_(buffer),
total_consumed_(0)
{
}

bool empty() const
{
return total_consumed_ >= buffer_.size();
}

Buffer prepare(std::size_t max_size)
{
return asio::buffer(buffer_ + total_consumed_, max_size);
}

void consume(std::size_t size)
{
total_consumed_ += size;
}

std::size_t total_consumed() const
{
return total_consumed_;
}

private:
Buffer buffer_;
std::size_t total_consumed_;
};

template <>
class consuming_buffers<mutable_buffer, mutable_buffer, const mutable_buffer*>
: public consuming_single_buffer<ASIO_MUTABLE_BUFFER>
{
public:
explicit consuming_buffers(const mutable_buffer& buffer)
: consuming_single_buffer<ASIO_MUTABLE_BUFFER>(buffer)
{
}
};

template <>
class consuming_buffers<const_buffer, mutable_buffer, const mutable_buffer*>
: public consuming_single_buffer<ASIO_CONST_BUFFER>
{
public:
explicit consuming_buffers(const mutable_buffer& buffer)
: consuming_single_buffer<ASIO_CONST_BUFFER>(buffer)
{
}
};

template <>
class consuming_buffers<const_buffer, const_buffer, const const_buffer*>
: public consuming_single_buffer<ASIO_CONST_BUFFER>
{
public:
explicit consuming_buffers(const const_buffer& buffer)
: consuming_single_buffer<ASIO_CONST_BUFFER>(buffer)
{
}
};

#if !defined(ASIO_NO_DEPRECATED)

template <>
class consuming_buffers<mutable_buffer,
mutable_buffers_1, const mutable_buffer*>
: public consuming_single_buffer<ASIO_MUTABLE_BUFFER>
{
public:
explicit consuming_buffers(const mutable_buffers_1& buffer)
: consuming_single_buffer<ASIO_MUTABLE_BUFFER>(buffer)
{
}
};

template <>
class consuming_buffers<const_buffer, mutable_buffers_1, const mutable_buffer*>
: public consuming_single_buffer<ASIO_CONST_BUFFER>
{
public:
explicit consuming_buffers(const mutable_buffers_1& buffer)
: consuming_single_buffer<ASIO_CONST_BUFFER>(buffer)
{
}
};

template <>
class consuming_buffers<const_buffer, const_buffers_1, const const_buffer*>
: public consuming_single_buffer<ASIO_CONST_BUFFER>
{
public:
explicit consuming_buffers(const const_buffers_1& buffer)
: consuming_single_buffer<ASIO_CONST_BUFFER>(buffer)
{
}
};

#endif 

template <typename Buffer, typename Elem>
class consuming_buffers<Buffer, boost::array<Elem, 2>,
typename boost::array<Elem, 2>::const_iterator>
{
public:
explicit consuming_buffers(const boost::array<Elem, 2>& buffers)
: buffers_(buffers),
total_consumed_(0)
{
}

bool empty() const
{
return total_consumed_ >=
Buffer(buffers_[0]).size() + Buffer(buffers_[1]).size();
}

boost::array<Buffer, 2> prepare(std::size_t max_size)
{
boost::array<Buffer, 2> result = {{
Buffer(buffers_[0]), Buffer(buffers_[1]) }};
std::size_t buffer0_size = result[0].size();
result[0] = asio::buffer(result[0] + total_consumed_, max_size);
result[1] = asio::buffer(
result[1] + (total_consumed_ < buffer0_size
? 0 : total_consumed_ - buffer0_size),
max_size - result[0].size());
return result;
}

void consume(std::size_t size)
{
total_consumed_ += size;
}

std::size_t total_consumed() const
{
return total_consumed_;
}

private:
boost::array<Elem, 2> buffers_;
std::size_t total_consumed_;
};

#if defined(ASIO_HAS_STD_ARRAY)

template <typename Buffer, typename Elem>
class consuming_buffers<Buffer, std::array<Elem, 2>,
typename std::array<Elem, 2>::const_iterator>
{
public:
explicit consuming_buffers(const std::array<Elem, 2>& buffers)
: buffers_(buffers),
total_consumed_(0)
{
}

bool empty() const
{
return total_consumed_ >=
Buffer(buffers_[0]).size() + Buffer(buffers_[1]).size();
}

std::array<Buffer, 2> prepare(std::size_t max_size)
{
std::array<Buffer, 2> result = {{
Buffer(buffers_[0]), Buffer(buffers_[1]) }};
std::size_t buffer0_size = result[0].size();
result[0] = asio::buffer(result[0] + total_consumed_, max_size);
result[1] = asio::buffer(
result[1] + (total_consumed_ < buffer0_size
? 0 : total_consumed_ - buffer0_size),
max_size - result[0].size());
return result;
}

void consume(std::size_t size)
{
total_consumed_ += size;
}

std::size_t total_consumed() const
{
return total_consumed_;
}

private:
std::array<Elem, 2> buffers_;
std::size_t total_consumed_;
};

#endif 

template <typename Buffer>
class consuming_buffers<Buffer, null_buffers, const mutable_buffer*>
: public asio::null_buffers
{
public:
consuming_buffers(const null_buffers&)
{
}

bool empty()
{
return false;
}

null_buffers prepare(std::size_t)
{
return null_buffers();
}

void consume(std::size_t)
{
}

std::size_t total_consumed() const
{
return 0;
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
