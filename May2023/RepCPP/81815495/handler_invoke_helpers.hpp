
#ifndef ASIO_DETAIL_HANDLER_INVOKE_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_INVOKE_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/handler_invoke_hook.hpp"

#include "asio/detail/push_options.hpp"

namespace asio_handler_invoke_helpers {

#if defined(ASIO_NO_DEPRECATED)
template <typename Function, typename Context>
inline void error_if_hook_is_defined(Function& function, Context& context)
{
using asio::asio_handler_invoke;
(void)static_cast<asio::asio_handler_invoke_is_no_longer_used>(
asio_handler_invoke(function, asio::detail::addressof(context)));
}
#endif 

template <typename Function, typename Context>
inline void invoke(Function& function, Context& context)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
Function tmp(function);
tmp();
#elif defined(ASIO_NO_DEPRECATED)
(void)&error_if_hook_is_defined<Function, Context>;
(void)context;
function();
#else
using asio::asio_handler_invoke;
asio_handler_invoke(function, asio::detail::addressof(context));
#endif
}

template <typename Function, typename Context>
inline void invoke(const Function& function, Context& context)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
Function tmp(function);
tmp();
#elif defined(ASIO_NO_DEPRECATED)
(void)&error_if_hook_is_defined<const Function, Context>;
(void)context;
Function tmp(function);
tmp();
#else
using asio::asio_handler_invoke;
asio_handler_invoke(function, asio::detail::addressof(context));
#endif
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
