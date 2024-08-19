
#ifndef ASIO_SSL_DETAIL_OPENSSL_TYPES_HPP
#define ASIO_SSL_DETAIL_OPENSSL_TYPES_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/socket_types.hpp"
#if defined(ASIO_USE_WOLFSSL)
# include <wolfssl/options.h>
#endif 
#include <openssl/conf.h>
#include <openssl/ssl.h>
#if !defined(OPENSSL_NO_ENGINE)
# include <openssl/engine.h>
#endif 
#include <openssl/dh.h>
#include <openssl/err.h>
#include <openssl/rsa.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#endif 
