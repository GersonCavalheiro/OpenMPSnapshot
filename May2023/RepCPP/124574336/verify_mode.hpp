
#ifndef BOOST_ASIO_SSL_VERIFY_MODE_HPP
#define BOOST_ASIO_SSL_VERIFY_MODE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/ssl/detail/openssl_types.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace ssl {


typedef int verify_mode;

#if defined(GENERATING_DOCUMENTATION)
const int verify_none = implementation_defined;

const int verify_peer = implementation_defined;

const int verify_fail_if_no_peer_cert = implementation_defined;

const int verify_client_once = implementation_defined;
#else
const int verify_none = SSL_VERIFY_NONE;
const int verify_peer = SSL_VERIFY_PEER;
const int verify_fail_if_no_peer_cert = SSL_VERIFY_FAIL_IF_NO_PEER_CERT;
const int verify_client_once = SSL_VERIFY_CLIENT_ONCE;
#endif

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
