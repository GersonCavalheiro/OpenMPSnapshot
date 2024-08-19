
#ifndef ASIO_IO_SERVICE_HPP
#define ASIO_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/io_context.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

#if !defined(ASIO_NO_DEPRECATED)
typedef io_context io_service;
#endif 

} 

#include "asio/detail/pop_options.hpp"

#endif 
