
#ifndef BOOST_ASIO_IO_SERVICE_HPP
#define BOOST_ASIO_IO_SERVICE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/io_context.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if !defined(BOOST_ASIO_NO_DEPRECATED)
typedef io_context io_service;
#endif 

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
