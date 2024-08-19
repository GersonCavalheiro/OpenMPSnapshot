
#ifndef BOOST_ASIO_DETAIL_SCHEDULER_OPERATION_HPP
#define BOOST_ASIO_DETAIL_SCHEDULER_OPERATION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/system/error_code.hpp>
#include <boost/asio/detail/handler_tracking.hpp>
#include <boost/asio/detail/op_queue.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class scheduler;

class scheduler_operation BOOST_ASIO_INHERIT_TRACKED_HANDLER
{
public:
typedef scheduler_operation operation_type;

void complete(void* owner, const boost::system::error_code& ec,
std::size_t bytes_transferred)
{
func_(owner, this, ec, bytes_transferred);
}

void destroy()
{
func_(0, this, boost::system::error_code(), 0);
}

protected:
typedef void (*func_type)(void*,
scheduler_operation*,
const boost::system::error_code&, std::size_t);

scheduler_operation(func_type func)
: next_(0),
func_(func),
task_result_(0)
{
}

~scheduler_operation()
{
}

private:
friend class op_queue_access;
scheduler_operation* next_;
func_type func_;
protected:
friend class scheduler;
unsigned int task_result_; 
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
