
#ifndef ASIO_DETAIL_SOLARIS_FENCED_BLOCK_HPP
#define ASIO_DETAIL_SOLARIS_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(__sun)

#include <atomic.h>
#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class solaris_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit solaris_fenced_block(half_t)
{
}

explicit solaris_fenced_block(full_t)
{
membar_consumer();
}

~solaris_fenced_block()
{
membar_producer();
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
