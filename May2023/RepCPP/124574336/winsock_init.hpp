
#ifndef BOOST_ASIO_DETAIL_WINSOCK_INIT_HPP
#define BOOST_ASIO_DETAIL_WINSOCK_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(BOOST_ASIO_WINDOWS) || defined(__CYGWIN__)

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class winsock_init_base
{
protected:
struct data
{
long init_count_;
long result_;
};

BOOST_ASIO_DECL static void startup(data& d,
unsigned char major, unsigned char minor);

BOOST_ASIO_DECL static void manual_startup(data& d);

BOOST_ASIO_DECL static void cleanup(data& d);

BOOST_ASIO_DECL static void manual_cleanup(data& d);

BOOST_ASIO_DECL static void throw_on_error(data& d);
};

template <int Major = 2, int Minor = 0>
class winsock_init : private winsock_init_base
{
public:
winsock_init(bool allow_throw = true)
{
startup(data_, Major, Minor);
if (allow_throw)
throw_on_error(data_);
}

winsock_init(const winsock_init&)
{
startup(data_, Major, Minor);
throw_on_error(data_);
}

~winsock_init()
{
cleanup(data_);
}

class manual
{
public:
manual()
{
manual_startup(data_);
}

manual(const manual&)
{
manual_startup(data_);
}

~manual()
{
manual_cleanup(data_);
}
};

private:
friend class manual;
static data data_;
};

template <int Major, int Minor>
winsock_init_base::data winsock_init<Major, Minor>::data_;

static const winsock_init<>& winsock_init_instance = winsock_init<>(false);

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/detail/impl/winsock_init.ipp>
#endif 

#endif 

#endif 
