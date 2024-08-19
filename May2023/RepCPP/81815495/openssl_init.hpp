
#ifndef ASIO_SSL_DETAIL_OPENSSL_INIT_HPP
#define ASIO_SSL_DETAIL_OPENSSL_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstring>
#include "asio/detail/memory.hpp"
#include "asio/detail/noncopyable.hpp"
#include "asio/ssl/detail/openssl_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

class openssl_init_base
: private noncopyable
{
protected:
class do_init;

ASIO_DECL static asio::detail::shared_ptr<do_init> instance();

#if !defined(SSL_OP_NO_COMPRESSION) \
&& (OPENSSL_VERSION_NUMBER >= 0x00908000L)
ASIO_DECL static STACK_OF(SSL_COMP)* get_null_compression_methods();
#endif 
};

template <bool Do_Init = true>
class openssl_init : private openssl_init_base
{
public:
openssl_init()
: ref_(instance())
{
using namespace std; 

openssl_init* tmp = &instance_;
memmove(&tmp, &tmp, sizeof(openssl_init*));
}

~openssl_init()
{
}

#if !defined(SSL_OP_NO_COMPRESSION) \
&& (OPENSSL_VERSION_NUMBER >= 0x00908000L)
using openssl_init_base::get_null_compression_methods;
#endif 

private:
static openssl_init instance_;

asio::detail::shared_ptr<do_init> ref_;
};

template <bool Do_Init>
openssl_init<Do_Init> openssl_init<Do_Init>::instance_;

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/detail/impl/openssl_init.ipp"
#endif 

#endif 
