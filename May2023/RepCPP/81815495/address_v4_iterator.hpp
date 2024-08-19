
#ifndef ASIO_IP_ADDRESS_V4_ITERATOR_HPP
#define ASIO_IP_ADDRESS_V4_ITERATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/ip/address_v4.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ip {

template <typename> class basic_address_iterator;


template <> class basic_address_iterator<address_v4>
{
public:
typedef address_v4 value_type;

typedef std::ptrdiff_t difference_type;

typedef const address_v4* pointer;

typedef const address_v4& reference;

typedef std::input_iterator_tag iterator_category;

basic_address_iterator(const address_v4& addr) ASIO_NOEXCEPT
: address_(addr)
{
}

basic_address_iterator(
const basic_address_iterator& other) ASIO_NOEXCEPT
: address_(other.address_)
{
}

#if defined(ASIO_HAS_MOVE)
basic_address_iterator(basic_address_iterator&& other) ASIO_NOEXCEPT
: address_(ASIO_MOVE_CAST(address_v4)(other.address_))
{
}
#endif 

basic_address_iterator& operator=(
const basic_address_iterator& other) ASIO_NOEXCEPT
{
address_ = other.address_;
return *this;
}

#if defined(ASIO_HAS_MOVE)
basic_address_iterator& operator=(
basic_address_iterator&& other) ASIO_NOEXCEPT
{
address_ = ASIO_MOVE_CAST(address_v4)(other.address_);
return *this;
}
#endif 

const address_v4& operator*() const ASIO_NOEXCEPT
{
return address_;
}

const address_v4* operator->() const ASIO_NOEXCEPT
{
return &address_;
}

basic_address_iterator& operator++() ASIO_NOEXCEPT
{
address_ = address_v4((address_.to_uint() + 1) & 0xFFFFFFFF);
return *this;
}

basic_address_iterator operator++(int) ASIO_NOEXCEPT
{
basic_address_iterator tmp(*this);
++*this;
return tmp;
}

basic_address_iterator& operator--() ASIO_NOEXCEPT
{
address_ = address_v4((address_.to_uint() - 1) & 0xFFFFFFFF);
return *this;
}

basic_address_iterator operator--(int)
{
basic_address_iterator tmp(*this);
--*this;
return tmp;
}

friend bool operator==(const basic_address_iterator& a,
const basic_address_iterator& b)
{
return a.address_ == b.address_;
}

friend bool operator!=(const basic_address_iterator& a,
const basic_address_iterator& b)
{
return a.address_ != b.address_;
}

private:
address_v4 address_;
};

typedef basic_address_iterator<address_v4> address_v4_iterator;

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
