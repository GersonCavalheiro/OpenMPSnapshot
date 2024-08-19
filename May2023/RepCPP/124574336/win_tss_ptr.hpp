
#ifndef BOOST_ASIO_DETAIL_WIN_TSS_PTR_HPP
#define BOOST_ASIO_DETAIL_WIN_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS)

#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/detail/socket_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

BOOST_ASIO_DECL DWORD win_tss_ptr_create();

template <typename T>
class win_tss_ptr
: private noncopyable
{
public:
win_tss_ptr()
: tss_key_(win_tss_ptr_create())
{
}

~win_tss_ptr()
{
::TlsFree(tss_key_);
}

operator T*() const
{
return static_cast<T*>(::TlsGetValue(tss_key_));
}

void operator=(T* value)
{
::TlsSetValue(tss_key_, value);
}

private:
DWORD tss_key_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/win_tss_ptr.ipp>
#endif 

#endif 

#endif 
