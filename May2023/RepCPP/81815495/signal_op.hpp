
#ifndef ASIO_DETAIL_SIGNAL_OP_HPP
#define ASIO_DETAIL_SIGNAL_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class signal_op
: public operation
{
public:
asio::error_code ec_;

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

#include "asio/detail/pop_options.hpp"

#endif 
