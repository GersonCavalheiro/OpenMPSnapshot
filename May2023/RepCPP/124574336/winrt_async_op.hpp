
#ifndef BOOST_ASIO_DETAIL_WINRT_ASYNC_OP_HPP
#define BOOST_ASIO_DETAIL_WINRT_ASYNC_OP_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/operation.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

template <typename TResult>
class winrt_async_op
: public operation
{
public:
boost::system::error_code ec_;

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
boost::system::error_code ec_;

protected:
winrt_async_op(func_type complete_func)
: operation(complete_func)
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
