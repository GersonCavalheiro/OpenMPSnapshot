
#ifndef BOOST_ASIO_DETAIL_KEYWORD_TSS_PTR_HPP
#define BOOST_ASIO_DETAIL_KEYWORD_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_THREAD_KEYWORD_EXTENSION)

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename T>
class keyword_tss_ptr
: private noncopyable
{
public:
keyword_tss_ptr()
{
}

~keyword_tss_ptr()
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
static BOOST_ASIO_THREAD_KEYWORD T* value_;
};

template <typename T>
BOOST_ASIO_THREAD_KEYWORD T* keyword_tss_ptr<T>::value_;

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
