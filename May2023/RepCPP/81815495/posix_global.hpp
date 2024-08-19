
#ifndef ASIO_DETAIL_POSIX_GLOBAL_HPP
#define ASIO_DETAIL_POSIX_GLOBAL_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include <exception>
#include <pthread.h>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename T>
struct posix_global_impl
{
static void do_init()
{
instance_.static_ptr_ = instance_.ptr_ = new T;
}

~posix_global_impl()
{
delete static_ptr_;
}

static ::pthread_once_t init_once_;
static T* static_ptr_;
static posix_global_impl instance_;
T* ptr_;
};

template <typename T>
::pthread_once_t posix_global_impl<T>::init_once_ = PTHREAD_ONCE_INIT;

template <typename T>
T* posix_global_impl<T>::static_ptr_ = 0;

template <typename T>
posix_global_impl<T> posix_global_impl<T>::instance_;

template <typename T>
T& posix_global()
{
int result = ::pthread_once(
&posix_global_impl<T>::init_once_,
&posix_global_impl<T>::do_init);

if (result != 0)
std::terminate();

return *posix_global_impl<T>::instance_.ptr_;
}

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
