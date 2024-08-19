
#ifndef ASIO_HANDLER_INVOKE_HOOK_HPP
#define ASIO_HANDLER_INVOKE_HOOK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {




#if defined(ASIO_NO_DEPRECATED)

enum asio_handler_invoke_is_no_longer_used {};

typedef asio_handler_invoke_is_no_longer_used
asio_handler_invoke_is_deprecated;

#else 

typedef void asio_handler_invoke_is_deprecated;

#endif 

template <typename Function>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(Function& function, ...)
{
function();
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}

template <typename Function>
inline asio_handler_invoke_is_deprecated
asio_handler_invoke(const Function& function, ...)
{
Function tmp(function);
tmp();
#if defined(ASIO_NO_DEPRECATED)
return asio_handler_invoke_is_no_longer_used();
#endif 
}



} 

#include "asio/detail/pop_options.hpp"

#endif 
