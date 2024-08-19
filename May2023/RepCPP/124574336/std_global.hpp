
#ifndef BOOST_ASIO_DETAIL_STD_GLOBAL_HPP
#define BOOST_ASIO_DETAIL_STD_GLOBAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_STD_CALL_ONCE)

#include <exception>
#include <mutex>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T>
struct std_global_impl
{
static void do_init()
{
instance_.ptr_ = new T;
}

~std_global_impl()
{
delete ptr_;
}

static std::once_flag init_once_;
static std_global_impl instance_;
T* ptr_;
};

template <typename T>
std::once_flag std_global_impl<T>::init_once_;

template <typename T>
std_global_impl<T> std_global_impl<T>::instance_;

template <typename T>
T& std_global()
{
std::call_once(std_global_impl<T>::init_once_, &std_global_impl<T>::do_init);
return *std_global_impl<T>::instance_.ptr_;
}

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
