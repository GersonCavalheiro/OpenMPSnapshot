
#ifndef ASIO_DETAIL_SIGNAL_INIT_HPP
#define ASIO_DETAIL_SIGNAL_INIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if !defined(ASIO_WINDOWS) && !defined(__CYGWIN__)

#include <csignal>

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <int Signal = SIGPIPE>
class signal_init
{
public:
signal_init()
{
std::signal(Signal, SIG_IGN);
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
