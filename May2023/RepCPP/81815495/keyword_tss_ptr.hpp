
#ifndef ASIO_DETAIL_KEYWORD_TSS_PTR_HPP
#define ASIO_DETAIL_KEYWORD_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_THREAD_KEYWORD_EXTENSION)

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

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
static ASIO_THREAD_KEYWORD T* value_;
};

template <typename T>
ASIO_THREAD_KEYWORD T* keyword_tss_ptr<T>::value_;

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
