
#ifndef BOOST_ASIO_DETAIL_WAIT_OP_HPP
#define BOOST_ASIO_DETAIL_WAIT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class wait_op
: public operation
{
public:
boost::system::error_code ec_;

protected:
wait_op(func_type func)
: operation(func)
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
