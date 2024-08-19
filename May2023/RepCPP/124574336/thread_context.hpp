
#ifndef BOOST_ASIO_DETAIL_THREAD_CONTEXT_HPP
#define BOOST_ASIO_DETAIL_THREAD_CONTEXT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <climits>
#include <cstddef>
#include <boost/asio/detail/call_stack.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class thread_info_base;

class thread_context
{
public:
typedef call_stack<thread_context, thread_info_base> thread_call_stack;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
