
#ifndef BOOST_ASIO_DETAIL_NULL_GLOBAL_HPP
#define BOOST_ASIO_DETAIL_NULL_GLOBAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T>
struct null_global_impl
{
null_global_impl()
: ptr_(0)
{
}

~null_global_impl()
{
delete ptr_;
}

static null_global_impl instance_;
T* ptr_;
};

template <typename T>
null_global_impl<T> null_global_impl<T>::instance_;

template <typename T>
T& null_global()
{
if (null_global_impl<T>::instance_.ptr_ == 0)
null_global_impl<T>::instance_.ptr_ = new T;
return *null_global_impl<T>::instance_.ptr_;
}

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
