
#ifndef ASIO_BASIC_STREAMBUF_HPP
#define ASIO_BASIC_STREAMBUF_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_IOSTREAM)

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <streambuf>
#include <vector>
#include "asio/basic_streambuf_fwd.hpp"
#include "asio/buffer.hpp"
#include "asio/detail/limits.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/detail/throw_exception.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {


#if defined(GENERATING_DOCUMENTATION)
template <typename Allocator = std::allocator<char> >
#else
template <typename Allocator>
#endif
class basic_streambuf
: public std::streambuf,
private noncopyable
{
public:
#if defined(GENERATING_DOCUMENTATION)
typedef implementation_defined const_buffers_type;

typedef implementation_defined mutable_buffers_type;
#else
typedef ASIO_CONST_BUFFER const_buffers_type;
typedef ASIO_MUTABLE_BUFFER mutable_buffers_type;
#endif


explicit basic_streambuf(
std::size_t maximum_size = (std::numeric_limits<std::size_t>::max)(),
const Allocator& allocator = Allocator())
: max_size_(maximum_size),
buffer_(allocator)
{
std::size_t pend = (std::min<std::size_t>)(max_size_, buffer_delta);
buffer_.resize((std::max<std::size_t>)(pend, 1));
setg(&buffer_[0], &buffer_[0], &buffer_[0]);
setp(&buffer_[0], &buffer_[0] + pend);
}


std::size_t size() const ASIO_NOEXCEPT
{
return pptr() - gptr();
}


std::size_t max_size() const ASIO_NOEXCEPT
{
return max_size_;
}


std::size_t capacity() const ASIO_NOEXCEPT
{
return buffer_.capacity();
}


const_buffers_type data() const ASIO_NOEXCEPT
{
return asio::buffer(asio::const_buffer(gptr(),
(pptr() - gptr()) * sizeof(char_type)));
}


mutable_buffers_type prepare(std::size_t n)
{
reserve(n);
return asio::buffer(asio::mutable_buffer(
pptr(), n * sizeof(char_type)));
}


void commit(std::size_t n)
{
n = std::min<std::size_t>(n, epptr() - pptr());
pbump(static_cast<int>(n));
setg(eback(), gptr(), pptr());
}


void consume(std::size_t n)
{
if (egptr() < pptr())
setg(&buffer_[0], gptr(), pptr());
if (gptr() + n > pptr())
n = pptr() - gptr();
gbump(static_cast<int>(n));
}

protected:
enum { buffer_delta = 128 };


int_type underflow()
{
if (gptr() < pptr())
{
setg(&buffer_[0], gptr(), pptr());
return traits_type::to_int_type(*gptr());
}
else
{
return traits_type::eof();
}
}


int_type overflow(int_type c)
{
if (!traits_type::eq_int_type(c, traits_type::eof()))
{
if (pptr() == epptr())
{
std::size_t buffer_size = pptr() - gptr();
if (buffer_size < max_size_ && max_size_ - buffer_size < buffer_delta)
{
reserve(max_size_ - buffer_size);
}
else
{
reserve(buffer_delta);
}
}

*pptr() = traits_type::to_char_type(c);
pbump(1);
return c;
}

return traits_type::not_eof(c);
}

void reserve(std::size_t n)
{
std::size_t gnext = gptr() - &buffer_[0];
std::size_t pnext = pptr() - &buffer_[0];
std::size_t pend = epptr() - &buffer_[0];

if (n <= pend - pnext)
{
return;
}

if (gnext > 0)
{
pnext -= gnext;
std::memmove(&buffer_[0], &buffer_[0] + gnext, pnext);
}

if (n > pend - pnext)
{
if (n <= max_size_ && pnext <= max_size_ - n)
{
pend = pnext + n;
buffer_.resize((std::max<std::size_t>)(pend, 1));
}
else
{
std::length_error ex("asio::streambuf too long");
asio::detail::throw_exception(ex);
}
}

setg(&buffer_[0], &buffer_[0], &buffer_[0] + pnext);
setp(&buffer_[0] + pnext, &buffer_[0] + pend);
}

private:
std::size_t max_size_;
std::vector<char_type, Allocator> buffer_;

friend std::size_t read_size_helper(
basic_streambuf& sb, std::size_t max_size)
{
return std::min<std::size_t>(
std::max<std::size_t>(512, sb.buffer_.capacity() - sb.size()),
std::min<std::size_t>(max_size, sb.max_size() - sb.size()));
}
};

#if defined(GENERATING_DOCUMENTATION)
template <typename Allocator = std::allocator<char> >
#else
template <typename Allocator>
#endif
class basic_streambuf_ref
{
public:
typedef typename basic_streambuf<Allocator>::const_buffers_type
const_buffers_type;

typedef typename basic_streambuf<Allocator>::mutable_buffers_type
mutable_buffers_type;

explicit basic_streambuf_ref(basic_streambuf<Allocator>& sb)
: sb_(sb)
{
}

basic_streambuf_ref(const basic_streambuf_ref& other) ASIO_NOEXCEPT
: sb_(other.sb_)
{
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
basic_streambuf_ref(basic_streambuf_ref&& other) ASIO_NOEXCEPT
: sb_(other.sb_)
{
}
#endif 

std::size_t size() const ASIO_NOEXCEPT
{
return sb_.size();
}

std::size_t max_size() const ASIO_NOEXCEPT
{
return sb_.max_size();
}

std::size_t capacity() const ASIO_NOEXCEPT
{
return sb_.capacity();
}

const_buffers_type data() const ASIO_NOEXCEPT
{
return sb_.data();
}

mutable_buffers_type prepare(std::size_t n)
{
return sb_.prepare(n);
}

void commit(std::size_t n)
{
return sb_.commit(n);
}

void consume(std::size_t n)
{
return sb_.consume(n);
}

private:
basic_streambuf<Allocator>& sb_;
};

} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
