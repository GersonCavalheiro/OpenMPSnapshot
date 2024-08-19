
#ifndef ASIO_DETAIL_SOCKET_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_SOCKET_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS_RUNTIME)

#if defined(ASIO_WINDOWS) \
|| defined(__CYGWIN__) \
|| defined(__SYMBIAN32__)

#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class socket_select_interrupter
{
public:
ASIO_DECL socket_select_interrupter();

ASIO_DECL ~socket_select_interrupter();

ASIO_DECL void recreate();

ASIO_DECL void interrupt();

ASIO_DECL bool reset();

socket_type read_descriptor() const
{
return read_descriptor_;
}

private:
ASIO_DECL void open_descriptors();

ASIO_DECL void close_descriptors();

socket_type read_descriptor_;

socket_type write_descriptor_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/socket_select_interrupter.ipp"
#endif 

#endif 

#endif 

#endif 
