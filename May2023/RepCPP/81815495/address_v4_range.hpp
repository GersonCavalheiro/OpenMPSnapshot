
#ifndef ASIO_IP_ADDRESS_V4_RANGE_HPP
#define ASIO_IP_ADDRESS_V4_RANGE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/ip/address_v4_iterator.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

template <typename> class basic_address_range;


template <> class basic_address_range<address_v4>
{
public:
typedef basic_address_iterator<address_v4> iterator;

basic_address_range() ASIO_NOEXCEPT
: begin_(address_v4()),
end_(address_v4())
{
}

explicit basic_address_range(const iterator& first,
const iterator& last) ASIO_NOEXCEPT
: begin_(first),
end_(last)
{
}

basic_address_range(const basic_address_range& other) ASIO_NOEXCEPT
: begin_(other.begin_),
end_(other.end_)
{
}

#if defined(ASIO_HAS_MOVE)
basic_address_range(basic_address_range&& other) ASIO_NOEXCEPT
: begin_(ASIO_MOVE_CAST(iterator)(other.begin_)),
end_(ASIO_MOVE_CAST(iterator)(other.end_))
{
}
#endif 

basic_address_range& operator=(
const basic_address_range& other) ASIO_NOEXCEPT
{
begin_ = other.begin_;
end_ = other.end_;
return *this;
}

#if defined(ASIO_HAS_MOVE)
basic_address_range& operator=(
basic_address_range&& other) ASIO_NOEXCEPT
{
begin_ = ASIO_MOVE_CAST(iterator)(other.begin_);
end_ = ASIO_MOVE_CAST(iterator)(other.end_);
return *this;
}
#endif 

iterator begin() const ASIO_NOEXCEPT
{
return begin_;
}

iterator end() const ASIO_NOEXCEPT
{
return end_;
}

bool empty() const ASIO_NOEXCEPT
{
return size() == 0;
}

std::size_t size() const ASIO_NOEXCEPT
{
return end_->to_uint() - begin_->to_uint();
}

iterator find(const address_v4& addr) const ASIO_NOEXCEPT
{
return addr >= *begin_ && addr < *end_ ? iterator(addr) : end_;
}

private:
iterator begin_;
iterator end_;
};

typedef basic_address_range<address_v4> address_v4_range;

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
