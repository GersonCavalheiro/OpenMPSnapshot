
#ifndef ASIO_SSL_DETAIL_ENGINE_HPP
#define ASIO_SSL_DETAIL_ENGINE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/buffer.hpp"
#include "asio/detail/static_mutex.hpp"
#include "asio/ssl/detail/openssl_types.hpp"
#include "asio/ssl/detail/verify_callback.hpp"
#include "asio/ssl/stream_base.hpp"
#include "asio/ssl/verify_mode.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace ssl {
namespace detail {

class engine
{
public:
enum want
{
want_input_and_retry = -2,

want_output_and_retry = -1,

want_nothing = 0,

want_output = 1
};

ASIO_DECL explicit engine(SSL_CTX* context);

#if defined(ASIO_HAS_MOVE)
ASIO_DECL engine(engine&& other) ASIO_NOEXCEPT;
#endif 

ASIO_DECL ~engine();

#if defined(ASIO_HAS_MOVE)
ASIO_DECL engine& operator=(engine&& other) ASIO_NOEXCEPT;
#endif 

ASIO_DECL SSL* native_handle();

ASIO_DECL asio::error_code set_verify_mode(
verify_mode v, asio::error_code& ec);

ASIO_DECL asio::error_code set_verify_depth(
int depth, asio::error_code& ec);

ASIO_DECL asio::error_code set_verify_callback(
verify_callback_base* callback, asio::error_code& ec);

ASIO_DECL want handshake(
stream_base::handshake_type type, asio::error_code& ec);

ASIO_DECL want shutdown(asio::error_code& ec);

ASIO_DECL want write(const asio::const_buffer& data,
asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL want read(const asio::mutable_buffer& data,
asio::error_code& ec, std::size_t& bytes_transferred);

ASIO_DECL asio::mutable_buffer get_output(
const asio::mutable_buffer& data);

ASIO_DECL asio::const_buffer put_input(
const asio::const_buffer& data);

ASIO_DECL const asio::error_code& map_error_code(
asio::error_code& ec) const;

private:
engine(const engine&);
engine& operator=(const engine&);

ASIO_DECL static int verify_callback_function(
int preverified, X509_STORE_CTX* ctx);

#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
ASIO_DECL static asio::detail::static_mutex& accept_mutex();
#endif 

ASIO_DECL want perform(int (engine::* op)(void*, std::size_t),
void* data, std::size_t length, asio::error_code& ec,
std::size_t* bytes_transferred);

ASIO_DECL int do_accept(void*, std::size_t);

ASIO_DECL int do_connect(void*, std::size_t);

ASIO_DECL int do_shutdown(void*, std::size_t);

ASIO_DECL int do_read(void* data, std::size_t length);

ASIO_DECL int do_write(void* data, std::size_t length);

SSL* ssl_;
BIO* ext_bio_;
};

} 
} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/ssl/detail/impl/engine.ipp"
#endif 

#endif 
