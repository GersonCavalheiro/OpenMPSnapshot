
#ifndef BOOST_ASIO_DETAIL_POSIX_TSS_PTR_HPP
#define BOOST_ASIO_DETAIL_POSIX_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_HAS_PTHREADS)

#include <pthread.h>
#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

BOOST_ASIO_DECL void posix_tss_ptr_create(pthread_key_t& key);

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
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/posix_tss_ptr.ipp>
#endif 

#endif 

#endif 
