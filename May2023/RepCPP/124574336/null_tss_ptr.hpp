
#ifndef BOOST_ASIO_DETAIL_NULL_TSS_PTR_HPP
#define BOOST_ASIO_DETAIL_NULL_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if !defined(BOOST_ASIO_HAS_THREADS)

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T>
class null_tss_ptr
: private noncopyable
{
public:
null_tss_ptr()
: value_(0)
{
}

~null_tss_ptr()
{
}

operator T*() const
{
return value_;
}

void operator=(T* value)
{
value_ = value;
}

private:
T* value_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
