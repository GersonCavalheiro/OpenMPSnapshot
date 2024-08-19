
#ifndef ASIO_IS_READ_BUFFERED_HPP
#define ASIO_IS_READ_BUFFERED_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/buffered_read_stream_fwd.hpp"
#include "asio/buffered_stream_fwd.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

namespace detail {

template <typename Stream>
char is_read_buffered_helper(buffered_stream<Stream>* s);

template <typename Stream>
char is_read_buffered_helper(buffered_read_stream<Stream>* s);

struct is_read_buffered_big_type { char data[10]; };
is_read_buffered_big_type is_read_buffered_helper(...);

} 

template <typename Stream>
class is_read_buffered
{
public:
#if defined(GENERATING_DOCUMENTATION)
static const bool value;
#else
ASIO_STATIC_CONSTANT(bool,
value = sizeof(detail::is_read_buffered_helper((Stream*)0)) == 1);
#endif
};

} 

#include "asio/detail/pop_options.hpp"

#endif 
