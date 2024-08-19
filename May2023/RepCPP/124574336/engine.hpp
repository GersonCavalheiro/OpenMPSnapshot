
#ifndef BOOST_ASIO_SSL_DETAIL_ENGINE_HPP
#define BOOST_ASIO_SSL_DETAIL_ENGINE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#include <boost/asio/buffer.hpp>
#include <boost/asio/detail/static_mutex.hpp>
#include <boost/asio/ssl/detail/openssl_types.hpp>
#include <boost/asio/ssl/detail/verify_callback.hpp>
#include <boost/asio/ssl/stream_base.hpp>
#include <boost/asio/ssl/verify_mode.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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

BOOST_ASIO_DECL explicit engine(SSL_CTX* context);

#if defined(BOOST_ASIO_HAS_MOVE)
BOOST_ASIO_DECL engine(engine&& other) BOOST_ASIO_NOEXCEPT;
#endif 

BOOST_ASIO_DECL ~engine();

BOOST_ASIO_DECL SSL* native_handle();

BOOST_ASIO_DECL boost::system::error_code set_verify_mode(
verify_mode v, boost::system::error_code& ec);

BOOST_ASIO_DECL boost::system::error_code set_verify_depth(
int depth, boost::system::error_code& ec);

BOOST_ASIO_DECL boost::system::error_code set_verify_callback(
verify_callback_base* callback, boost::system::error_code& ec);

BOOST_ASIO_DECL want handshake(
stream_base::handshake_type type, boost::system::error_code& ec);

BOOST_ASIO_DECL want shutdown(boost::system::error_code& ec);

BOOST_ASIO_DECL want write(const boost::asio::const_buffer& data,
boost::system::error_code& ec, std::size_t& bytes_transferred);

BOOST_ASIO_DECL want read(const boost::asio::mutable_buffer& data,
boost::system::error_code& ec, std::size_t& bytes_transferred);

BOOST_ASIO_DECL boost::asio::mutable_buffer get_output(
const boost::asio::mutable_buffer& data);

BOOST_ASIO_DECL boost::asio::const_buffer put_input(
const boost::asio::const_buffer& data);

BOOST_ASIO_DECL const boost::system::error_code& map_error_code(
boost::system::error_code& ec) const;

private:
engine(const engine&);
engine& operator=(const engine&);

BOOST_ASIO_DECL static int verify_callback_function(
int preverified, X509_STORE_CTX* ctx);

#if (OPENSSL_VERSION_NUMBER < 0x10000000L)
BOOST_ASIO_DECL static boost::asio::detail::static_mutex& accept_mutex();
#endif 

BOOST_ASIO_DECL want perform(int (engine::* op)(void*, std::size_t),
void* data, std::size_t length, boost::system::error_code& ec,
std::size_t* bytes_transferred);

BOOST_ASIO_DECL int do_accept(void*, std::size_t);

BOOST_ASIO_DECL int do_connect(void*, std::size_t);

BOOST_ASIO_DECL int do_shutdown(void*, std::size_t);

BOOST_ASIO_DECL int do_read(void* data, std::size_t length);

BOOST_ASIO_DECL int do_write(void* data, std::size_t length);

SSL* ssl_;
BIO* ext_bio_;
};

} 
} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/ssl/detail/impl/engine.ipp>
#endif 

#endif 
