
#ifndef BOOST_ASIO_SSL_DETAIL_OPENSSL_INIT_HPP
#define BOOST_ASIO_SSL_DETAIL_OPENSSL_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstring>
#include <boost/asio/detail/memory.hpp>
#include <boost/asio/detail/noncopyable.hpp>
#include <boost/asio/ssl/detail/openssl_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ssl {
namespace detail {

class openssl_init_base
: private noncopyable
{
protected:
class do_init;

BOOST_ASIO_DECL static boost::asio::detail::shared_ptr<do_init> instance();

#if !defined(SSL_OP_NO_COMPRESSION) \
&& (OPENSSL_VERSION_NUMBER >= 0x00908000L)
BOOST_ASIO_DECL static STACK_OF(SSL_COMP)* get_null_compression_methods();
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

boost::asio::detail::shared_ptr<do_init> ref_;
};

template <bool Do_Init>
openssl_init<Do_Init> openssl_init<Do_Init>::instance_;

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/ssl/detail/impl/openssl_init.ipp>
#endif 

#endif 
