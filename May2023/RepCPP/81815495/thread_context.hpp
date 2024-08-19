
#ifndef ASIO_DETAIL_THREAD_CONTEXT_HPP
#define ASIO_DETAIL_THREAD_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <climits>
#include <cstddef>
#include "asio/detail/call_stack.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class thread_info_base;

class thread_context
{
public:
ASIO_DECL static thread_info_base* top_of_thread_call_stack();

protected:
typedef call_stack<thread_context, thread_info_base> thread_call_stack;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#if defined(ASIO_HEADER_ONLY)
# include "asio/detail/impl/thread_context.ipp"
#endif 

#endif 
