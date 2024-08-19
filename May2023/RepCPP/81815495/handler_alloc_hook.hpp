
#ifndef ASIO_HANDLER_ALLOC_HOOK_HPP
#define ASIO_HANDLER_ALLOC_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include <cstddef>

#include "asio/detail/push_options.hpp"

namespace asio {

#if defined(ASIO_NO_DEPRECATED)

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


ASIO_DECL asio_handler_allocate_is_deprecated
asio_handler_allocate(std::size_t size, ...);


ASIO_DECL asio_handler_deallocate_is_deprecated
asio_handler_deallocate(void* pointer, std::size_t size, ...);

} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/impl/handler_alloc_hook.ipp"
#endif 

#endif 
