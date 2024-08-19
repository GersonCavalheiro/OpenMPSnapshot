
#ifndef BOOST_ASIO_DETAIL_REACTOR_OP_HPP
#define BOOST_ASIO_DETAIL_REACTOR_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class reactor_op
: public operation
{
public:
boost::system::error_code ec_;

std::size_t bytes_transferred_;

enum status { not_done, done, done_and_exhausted };

status perform()
{
return perform_func_(this);
}

protected:
typedef status (*perform_func_type)(reactor_op*);

reactor_op(const boost::system::error_code& success_ec,
perform_func_type perform_func, func_type complete_func)
: operation(complete_func),
ec_(success_ec),
bytes_transferred_(0),
perform_func_(perform_func)
{
}

private:
perform_func_type perform_func_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
