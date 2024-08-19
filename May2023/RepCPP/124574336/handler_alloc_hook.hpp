
#ifndef BOOST_ASIO_HANDLER_ALLOC_HOOK_HPP
#define BOOST_ASIO_HANDLER_ALLOC_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <cstddef>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {

#if defined(BOOST_ASIO_NO_DEPRECATED)

enum asio_handler_allocate_is_no_longer_used {};
enum asio_handler_deallocate_is_no_longer_used {};

typedef asio_handler_allocate_is_no_longer_used
asio_handler_allocate_is_deprecated;
typedef asio_handler_deallocate_is_no_longer_used
asio_handler_deallocate_is_deprecated;

#else 

typedef void* asio_handler_allocate_is_deprecated;
typedef void asio_handler_deallocate_is_deprecated;

#endif 


BOOST_ASIO_DECL asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size, ...);


BOOST_ASIO_DECL asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size, ...);

} 
} 

#include <boost/asio/detail/pop_options.hpp>

#if defined(BOOST_ASIO_HEADER_ONLY)
# include <boost/asio/impl/handler_alloc_hook.ipp>
#endif 

#endif 
