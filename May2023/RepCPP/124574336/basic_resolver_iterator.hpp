
#ifndef BOOST_ASIO_IP_BASIC_RESOLVER_ITERATOR_HPP
#define BOOST_ASIO_IP_BASIC_RESOLVER_ITERATOR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/socket_ops.hpp>
#include <boost/asio/detail/socket_types.hpp>
#include <boost/asio/ip/basic_resolver_entry.hpp>

#if defined(BOOST_ASIO_WINDOWS_RUNTIME)
# include <boost/asio/detail/winrt_utils.hpp>
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ip {


template <typename InternetProtocol>
class basic_resolver_iterator
{
public:
typedef std::ptrdiff_t difference_type;

typedef basic_resolver_entry<InternetProtocol> value_type;

typedef const basic_resolver_entry<InternetProtocol>* pointer;

typedef const basic_resolver_entry<InternetProtocol>& reference;

typedef std::forward_iterator_tag iterator_category;

basic_resolver_iterator()
: index_(0)
{
}

basic_resolver_iterator(const basic_resolver_iterator& other)
: values_(other.values_),
index_(other.index_)
{
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
basic_resolver_iterator(basic_resolver_iterator&& other)
: values_(BOOST_ASIO_MOVE_CAST(values_ptr_type)(other.values_)),
index_(other.index_)
{
other.index_ = 0;
}
#endif 

basic_resolver_iterator& operator=(const basic_resolver_iterator& other)
{
values_ = other.values_;
index_ = other.index_;
return *this;
}

#if defined(BOOST_ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
basic_resolver_iterator& operator=(basic_resolver_iterator&& other)
{
if (this != &other)
{
values_ = BOOST_ASIO_MOVE_CAST(values_ptr_type)(other.values_);
index_ = other.index_;
other.index_ = 0;
}

return *this;
}
#endif 

const basic_resolver_entry<InternetProtocol>& operator*() const
{
return dereference();
}

const basic_resolver_entry<InternetProtocol>* operator->() const
{
return &dereference();
}

basic_resolver_iterator& operator++()
{
increment();
return *this;
}

basic_resolver_iterator operator++(int)
{
basic_resolver_iterator tmp(*this);
++*this;
return tmp;
}

friend bool operator==(const basic_resolver_iterator& a,
const basic_resolver_iterator& b)
{
return a.equal(b);
}

friend bool operator!=(const basic_resolver_iterator& a,
const basic_resolver_iterator& b)
{
return !a.equal(b);
}

protected:
void increment()
{
if (++index_ == values_->size())
{
values_.reset();
index_ = 0;
}
}

bool equal(const basic_resolver_iterator& other) const
{
if (!values_ && !other.values_)
return true;
if (values_ != other.values_)
return false;
return index_ == other.index_;
}

const basic_resolver_entry<InternetProtocol>& dereference() const
{
return (*values_)[index_];
}

typedef std::vector<basic_resolver_entry<InternetProtocol> > values_type;
typedef boost::asio::detail::shared_ptr<values_type> values_ptr_type;
values_ptr_type values_;
std::size_t index_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
