
#ifndef BOOST_ASIO_DETAIL_NULL_FENCED_BLOCK_HPP
#define BOOST_ASIO_DETAIL_NULL_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class null_fenced_block
: private noncopyable
{
public:
enum half_or_full_t { half, full };

explicit null_fenced_block(half_or_full_t)
{
}

~null_fenced_block()
{
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
