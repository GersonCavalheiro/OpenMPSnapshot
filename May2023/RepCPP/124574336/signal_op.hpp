
#ifndef BOOST_ASIO_DETAIL_SIGNAL_OP_HPP
#define BOOST_ASIO_DETAIL_SIGNAL_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class signal_op
: public operation
{
public:
boost::system::error_code ec_;

int signal_number_;

protected:
signal_op(func_type func)
: operation(func),
signal_number_(0)
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
