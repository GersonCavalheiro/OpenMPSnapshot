
#ifndef ASIO_IMPL_SYSTEM_CONTEXT_HPP
#define ASIO_IMPL_SYSTEM_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/system_executor.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {

inline system_context::executor_type
system_context::get_executor() ASIO_NOEXCEPT
{
return system_executor();
}

} 

#include "asio/detail/pop_options.hpp"

#endif 
