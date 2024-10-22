
#ifndef ASIO_DETAIL_PIPE_SELECT_INTERRUPTER_HPP
#define ASIO_DETAIL_PIPE_SELECT_INTERRUPTER_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS)
#if !defined(ASIO_WINDOWS_RUNTIME)
#if !defined(__CYGWIN__)
#if !defined(__SYMBIAN32__)
#if !defined(ASIO_HAS_EVENTFD)

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class pipe_select_interrupter
{
public:
ASIO_DECL pipe_select_interrupter();

ASIO_DECL ~pipe_select_interrupter();

ASIO_DECL void recreate();

ASIO_DECL void interrupt();

ASIO_DECL bool reset();

int read_descriptor() const
{
return read_descriptor_;
}

private:
ASIO_DECL void open_descriptors();

ASIO_DECL void close_descriptors();

int read_descriptor_;

int write_descriptor_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/pipe_select_interrupter.ipp"
#endif 

#endif 
#endif 
#endif 
#endif 
#endif 

#endif 
