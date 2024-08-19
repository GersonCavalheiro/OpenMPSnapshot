
#ifndef ASIO_DETAIL_WAIT_OP_HPP
#define ASIO_DETAIL_WAIT_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class wait_op
: public operation
{
public:
asio::error_code ec_;

void* cancellation_key_;

protected:
wait_op(func_type func)
: operation(func),
cancellation_key_(0)
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
