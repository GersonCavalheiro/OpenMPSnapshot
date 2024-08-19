
#ifndef BOOST_ASIO_DETAIL_STD_THREAD_HPP
#define BOOST_ASIO_DETAIL_STD_THREAD_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_STD_THREAD)

#include <thread>
#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class std_thread
: private noncopyable
{
public:
template <typename Function>
std_thread(Function f, unsigned int = 0)
: thread_(f)
{
}

~std_thread()
{
join();
}

void join()
{
if (thread_.joinable())
thread_.join();
}

static std::size_t hardware_concurrency()
{
return std::thread::hardware_concurrency();
}

private:
std::thread thread_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
