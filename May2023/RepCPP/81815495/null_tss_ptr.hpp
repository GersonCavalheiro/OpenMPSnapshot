
#ifndef ASIO_DETAIL_NULL_TSS_PTR_HPP
#define ASIO_DETAIL_NULL_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_HAS_THREADS)

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

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

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
