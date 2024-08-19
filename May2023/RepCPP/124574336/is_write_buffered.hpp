
#ifndef BOOST_ASIO_IS_WRITE_BUFFERED_HPP
#define BOOST_ASIO_IS_WRITE_BUFFERED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/buffered_stream_fwd.hpp>
#include <boost/asio/buffered_write_stream_fwd.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

namespace detail {

template <typename Stream>
char is_write_buffered_helper(buffered_stream<Stream>* s);

template <typename Stream>
char is_write_buffered_helper(buffered_write_stream<Stream>* s);

struct is_write_buffered_big_type { char data[10]; };
is_write_buffered_big_type is_write_buffered_helper(...);

} 

template <typename Stream>
class is_write_buffered
{
public:
#if defined(GENERATING_DOCUMENTATION)
static const bool value;
#else
BOOST_ASIO_STATIC_CONSTANT(bool,
value = sizeof(detail::is_write_buffered_helper((Stream*)0)) == 1);
#endif
};

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
