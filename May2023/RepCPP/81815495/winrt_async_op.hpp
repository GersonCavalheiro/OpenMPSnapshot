
#ifndef ASIO_DETAIL_WINRT_ASYNC_OP_HPP
#define ASIO_DETAIL_WINRT_ASYNC_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/operation.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

template <typename TResult>
class winrt_async_op
: public operation
{
public:
asio::error_code ec_;

TResult result_;

protected:
winrt_async_op(func_type complete_func)
: operation(complete_func),
result_()
{
}
};

template <>
class winrt_async_op<void>
: public operation
{
public:
asio::error_code ec_;

protected:
winrt_async_op(func_type complete_func)
: operation(complete_func)
{
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
