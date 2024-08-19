
#ifndef ASIO_DETAIL_WIN_TSS_PTR_HPP
#define ASIO_DETAIL_WIN_TSS_PTR_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

ASIO_DECL DWORD win_tss_ptr_create();

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

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/win_tss_ptr.ipp"
#endif 

#endif 

#endif 
