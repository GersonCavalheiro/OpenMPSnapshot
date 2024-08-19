
#ifndef ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP
#define ASIO_DETAIL_HANDLER_CONT_HELPERS_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/memory.hpp"
#include "asio/handler_continuation_hook.hpp"

#include "asio/detail/push_options.hpp"

namespace asio_handler_cont_helpers {

template <typename Context>
inline bool is_continuation(Context& context)
{
#if !defined(ASIO_HAS_HANDLER_HOOKS)
return false;
#else
using asio::asio_handler_is_continuation;
return asio_handler_is_continuation(
asio::detail::addressof(context));
#endif
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
