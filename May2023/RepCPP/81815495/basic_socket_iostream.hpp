
#ifndef ASIO_BASIC_SOCKET_IOSTREAM_HPP
#define ASIO_BASIC_SOCKET_IOSTREAM_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_NO_IOSTREAM)

#include <istream>
#include <ostream>
#include "asio/basic_socket_streambuf.hpp"

#if !defined(ASIO_HAS_VARIADIC_TEMPLATES)

# include "asio/detail/variadic_templates.hpp"


# define ASIO_PRIVATE_CTR_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
explicit basic_socket_iostream(ASIO_VARIADIC_BYVAL_PARAMS(n)) \
: std::basic_iostream<char>( \
&this->detail::socket_iostream_base< \
Protocol, Clock, WaitTraits>::streambuf_) \
{ \
this->setf(std::ios_base::unitbuf); \
if (rdbuf()->connect(ASIO_VARIADIC_BYVAL_ARGS(n)) == 0) \
this->setstate(std::ios_base::failbit); \
} \



# define ASIO_PRIVATE_CONNECT_DEF(n) \
template <ASIO_VARIADIC_TPARAMS(n)> \
void connect(ASIO_VARIADIC_BYVAL_PARAMS(n)) \
{ \
if (rdbuf()->connect(ASIO_VARIADIC_BYVAL_ARGS(n)) == 0) \
this->setstate(std::ios_base::failbit); \
} \


#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename Protocol, typename Clock, typename WaitTraits>
class socket_iostream_base
{
protected:
socket_iostream_base()
{
}

#if defined(ASIO_HAS_MOVE)
socket_iostream_base(socket_iostream_base&& other)
: streambuf_(std::move(other.streambuf_))
{
}

socket_iostream_base(basic_stream_socket<Protocol> s)
: streambuf_(std::move(s))
{
}

socket_iostream_base& operator=(socket_iostream_base&& other)
{
streambuf_ = std::move(other.streambuf_);
return *this;
}
#endif 

basic_socket_streambuf<Protocol, Clock, WaitTraits> streambuf_;
};

} 

#if !defined(ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL)
#define ASIO_BASIC_SOCKET_IOSTREAM_FWD_DECL

template <typename Protocol,
#if defined(ASIO_HAS_BOOST_DATE_TIME) \
&& defined(ASIO_USE_BOOST_DATE_TIME_FOR_SOCKET_IOSTREAM)
typename Clock = boost::posix_time::ptime,
typename WaitTraits = time_traits<Clock> >
#else 
typename Clock = chrono::steady_clock,
typename WaitTraits = wait_traits<Clock> >
#endif 
class basic_socket_iostream;

#endif 

#if defined(GENERATING_DOCUMENTATION)
template <typename Protocol,
typename Clock = chrono::steady_clock,
typename WaitTraits = wait_traits<Clock> >
#else 
template <typename Protocol, typename Clock, typename WaitTraits>
#endif 
class basic_socket_iostream
: private detail::socket_iostream_base<Protocol, Clock, WaitTraits>,
public std::basic_iostream<char>
{
private:
#if defined(ASIO_HAS_BOOST_DATE_TIME) \
&& defined(ASIO_USE_BOOST_DATE_TIME_FOR_SOCKET_IOSTREAM)
typedef WaitTraits traits_helper;
#else 
typedef detail::chrono_time_traits<Clock, WaitTraits> traits_helper;
#endif 

public:
typedef Protocol protocol_type;

typedef typename Protocol::endpoint endpoint_type;

typedef Clock clock_type;

#if defined(GENERATING_DOCUMENTATION)
typedef typename WaitTraits::time_type time_type;

typedef typename WaitTraits::time_point time_point;

typedef typename WaitTraits::duration_type duration_type;

typedef typename WaitTraits::duration duration;
#else
# if !defined(ASIO_NO_DEPRECATED)
typedef typename traits_helper::time_type time_type;
typedef typename traits_helper::duration_type duration_type;
# endif 
typedef typename traits_helper::time_type time_point;
typedef typename traits_helper::duration_type duration;
#endif

basic_socket_iostream()
: std::basic_iostream<char>(
&this->detail::socket_iostream_base<
Protocol, Clock, WaitTraits>::streambuf_)
{
this->setf(std::ios_base::unitbuf);
}

#if defined(ASIO_HAS_MOVE) || defined(GENERATING_DOCUMENTATION)
explicit basic_socket_iostream(basic_stream_socket<protocol_type> s)
: detail::socket_iostream_base<
Protocol, Clock, WaitTraits>(std::move(s)),
std::basic_iostream<char>(
&this->detail::socket_iostream_base<
Protocol, Clock, WaitTraits>::streambuf_)
{
this->setf(std::ios_base::unitbuf);
}

#if defined(ASIO_HAS_STD_IOSTREAM_MOVE) \
|| defined(GENERATING_DOCUMENTATION)
basic_socket_iostream(basic_socket_iostream&& other)
: detail::socket_iostream_base<
Protocol, Clock, WaitTraits>(std::move(other)),
std::basic_iostream<char>(std::move(other))
{
this->set_rdbuf(&this->detail::socket_iostream_base<
Protocol, Clock, WaitTraits>::streambuf_);
}

basic_socket_iostream& operator=(basic_socket_iostream&& other)
{
std::basic_iostream<char>::operator=(std::move(other));
detail::socket_iostream_base<
Protocol, Clock, WaitTraits>::operator=(std::move(other));
return *this;
}
#endif 
#endif 

#if defined(GENERATING_DOCUMENTATION)

template <typename T1, ..., typename TN>
explicit basic_socket_iostream(T1 t1, ..., TN tn);
#elif defined(ASIO_HAS_VARIADIC_TEMPLATES)
template <typename... T>
explicit basic_socket_iostream(T... x)
: std::basic_iostream<char>(
&this->detail::socket_iostream_base<
Protocol, Clock, WaitTraits>::streambuf_)
{
this->setf(std::ios_base::unitbuf);
if (rdbuf()->connect(x...) == 0)
this->setstate(std::ios_base::failbit);
}
#else
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CTR_DEF)
#endif

#if defined(GENERATING_DOCUMENTATION)

template <typename T1, ..., typename TN>
void connect(T1 t1, ..., TN tn);
#elif defined(ASIO_HAS_VARIADIC_TEMPLATES)
template <typename... T>
void connect(T... x)
{
if (rdbuf()->connect(x...) == 0)
this->setstate(std::ios_base::failbit);
}
#else
ASIO_VARIADIC_GENERATE(ASIO_PRIVATE_CONNECT_DEF)
#endif

void close()
{
if (rdbuf()->close() == 0)
this->setstate(std::ios_base::failbit);
}

basic_socket_streambuf<Protocol, Clock, WaitTraits>* rdbuf() const
{
return const_cast<basic_socket_streambuf<Protocol, Clock, WaitTraits>*>(
&this->detail::socket_iostream_base<
Protocol, Clock, WaitTraits>::streambuf_);
}

basic_socket<Protocol>& socket()
{
return rdbuf()->socket();
}


const asio::error_code& error() const
{
return rdbuf()->error();
}

#if !defined(ASIO_NO_DEPRECATED)

time_point expires_at() const
{
return rdbuf()->expires_at();
}
#endif 


time_point expiry() const
{
return rdbuf()->expiry();
}


void expires_at(const time_point& expiry_time)
{
rdbuf()->expires_at(expiry_time);
}


void expires_after(const duration& expiry_time)
{
rdbuf()->expires_after(expiry_time);
}

#if !defined(ASIO_NO_DEPRECATED)

duration expires_from_now() const
{
return rdbuf()->expires_from_now();
}


void expires_from_now(const duration& expiry_time)
{
rdbuf()->expires_from_now(expiry_time);
}
#endif 

private:
basic_socket_iostream(const basic_socket_iostream&) ASIO_DELETED;
basic_socket_iostream& operator=(
const basic_socket_iostream&) ASIO_DELETED;
};

} 

#include "asio/detail/pop_options.hpp"

#if !defined(ASIO_HAS_VARIADIC_TEMPLATES)
# undef ASIO_PRIVATE_CTR_DEF
# undef ASIO_PRIVATE_CONNECT_DEF
#endif 

#endif 

#endif 
