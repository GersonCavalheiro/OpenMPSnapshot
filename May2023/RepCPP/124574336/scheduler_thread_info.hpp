
#ifndef BOOST_ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP
#define BOOST_ASIO_DETAIL_SCHEDULER_THREAD_INFO_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/op_queue.hpp>
#include <boost/asio/detail/thread_info_base.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
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
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
