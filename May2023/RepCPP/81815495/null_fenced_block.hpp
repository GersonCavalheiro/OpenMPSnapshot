
#ifndef ASIO_DETAIL_NULL_FENCED_BLOCK_HPP
#define ASIO_DETAIL_NULL_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

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

#include "asio/detail/pop_options.hpp"

#endif 
