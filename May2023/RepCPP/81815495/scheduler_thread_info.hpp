
#ifndef ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP
#define ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/op_queue.hpp"
#include "asio/detail/thread_info_base.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class scheduler;
class scheduler_operation;

struct scheduler_thread_info : public thread_info_base
{
op_queue<scheduler_operation> private_op_queue;
long private_outstanding_work;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
