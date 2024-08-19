
#ifndef ASIO_SSL_CONTEXT_BASE_HPP
#define ASIO_SSL_CONTEXT_BASE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/ssl/detail/openssl_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {

class context_base
{
public:
enum method
{
sslv2,

sslv2_client,

sslv2_server,

sslv3,

sslv3_client,

sslv3_server,

tlsv1,

tlsv1_client,

tlsv1_server,

sslv23,

sslv23_client,

sslv23_server,

tlsv11,

tlsv11_client,

tlsv11_server,

tlsv12,

tlsv12_client,

tlsv12_server,

tlsv13,

tlsv13_client,

tlsv13_server,

tls,

tls_client,

tls_server
};

typedef long options;

#if defined(GENERATING_DOCUMENTATION)
static const long default_workarounds = implementation_defined;

static const long single_dh_use = implementation_defined;

static const long no_sslv2 = implementation_defined;

static const long no_sslv3 = implementation_defined;

static const long no_tlsv1 = implementation_defined;

static const long no_tlsv1_1 = implementation_defined;

static const long no_tlsv1_2 = implementation_defined;

static const long no_tlsv1_3 = implementation_defined;

static const long no_compression = implementation_defined;
#else
ASIO_STATIC_CONSTANT(long, default_workarounds = SSL_OP_ALL);
ASIO_STATIC_CONSTANT(long, single_dh_use = SSL_OP_SINGLE_DH_USE);
ASIO_STATIC_CONSTANT(long, no_sslv2 = SSL_OP_NO_SSLv2);
ASIO_STATIC_CONSTANT(long, no_sslv3 = SSL_OP_NO_SSLv3);
ASIO_STATIC_CONSTANT(long, no_tlsv1 = SSL_OP_NO_TLSv1);
# if defined(SSL_OP_NO_TLSv1_1)
ASIO_STATIC_CONSTANT(long, no_tlsv1_1 = SSL_OP_NO_TLSv1_1);
# else 
ASIO_STATIC_CONSTANT(long, no_tlsv1_1 = 0x10000000L);
# endif 
# if defined(SSL_OP_NO_TLSv1_2)
ASIO_STATIC_CONSTANT(long, no_tlsv1_2 = SSL_OP_NO_TLSv1_2);
# else 
ASIO_STATIC_CONSTANT(long, no_tlsv1_2 = 0x08000000L);
# endif 
# if defined(SSL_OP_NO_TLSv1_3)
ASIO_STATIC_CONSTANT(long, no_tlsv1_3 = SSL_OP_NO_TLSv1_3);
# else 
ASIO_STATIC_CONSTANT(long, no_tlsv1_3 = 0x20000000L);
# endif 
# if defined(SSL_OP_NO_COMPRESSION)
ASIO_STATIC_CONSTANT(long, no_compression = SSL_OP_NO_COMPRESSION);
# else 
ASIO_STATIC_CONSTANT(long, no_compression = 0x20000L);
# endif 
#endif

enum file_format
{
asn1,

pem
};

#if !defined(GENERATING_DOCUMENTATION)
typedef int verify_mode;
ASIO_STATIC_CONSTANT(int, verify_none = SSL_VERIFY_NONE);
ASIO_STATIC_CONSTANT(int, verify_peer = SSL_VERIFY_PEER);
ASIO_STATIC_CONSTANT(int,
verify_fail_if_no_peer_cert = SSL_VERIFY_FAIL_IF_NO_PEER_CERT);
ASIO_STATIC_CONSTANT(int, verify_client_once = SSL_VERIFY_CLIENT_ONCE);
#endif

enum password_purpose
{
for_reading,

for_writing
};

protected:
~context_base()
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
