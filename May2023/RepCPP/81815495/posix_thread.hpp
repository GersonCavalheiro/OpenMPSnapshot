
#ifndef ASIO_DETAIL_POSIX_THREAD_HPP
#define ASIO_DETAIL_POSIX_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include <cstddef>
#include <pthread.h>
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

extern "C"
{
ASIO_DECL void* asio_detail_posix_thread_function(void* arg);
}

class posix_thread
: private noncopyable
{
public:
template <typename Function>
posix_thread(Function f, unsigned int = 0)
: joined_(false)
{
start_thread(new func<Function>(f));
}

ASIO_DECL ~posix_thread();

ASIO_DECL void join();

ASIO_DECL static std::size_t hardware_concurrency();

private:
friend void* asio_detail_posix_thread_function(void* arg);

class func_base
{
public:
virtual ~func_base() {}
virtual void run() = 0;
};

struct auto_func_base_ptr
{
func_base* ptr;
~auto_func_base_ptr() { delete ptr; }
};

template <typename Function>
class func
: public func_base
{
public:
func(Function f)
: f_(f)
{
}

virtual void run()
{
f_();
}

private:
Function f_;
};

ASIO_DECL void start_thread(func_base* arg);

::pthread_t thread_;
bool joined_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/posix_thread.ipp"
#endif 

#endif 

#endif 
