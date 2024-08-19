
#ifndef ASIO_DETAIL_POSIX_TSS_PTR_HPP
#define ASIO_DETAIL_POSIX_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_HAS_PTHREADS)

#include <pthread.h>
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

ASIO_DECL void posix_tss_ptr_create(pthread_key_t& key);

template <typename T>
class posix_tss_ptr
: private noncopyable
{
public:
posix_tss_ptr()
{
posix_tss_ptr_create(tss_key_);
}

~posix_tss_ptr()
{
::pthread_key_delete(tss_key_);
}

operator T*() const
{
return static_cast<T*>(::pthread_getspecific(tss_key_));
}

void operator=(T* value)
{
::pthread_setspecific(tss_key_, value);
}

private:
pthread_key_t tss_key_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/posix_tss_ptr.ipp"
#endif 

#endif 

#endif 
