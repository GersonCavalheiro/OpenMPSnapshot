
#ifndef ASIO_DETAIL_LOCAL_FREE_ON_BLOCK_EXIT_HPP
#define ASIO_DETAIL_LOCAL_FREE_ON_BLOCK_EXIT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(ASIO_WINDOWS) || defined(__CYGWIN__)
#if !defined(ASIO_WINDOWS_APP)

#include "asio/detail/noncopyable.hpp"
#include "asio/detail/socket_types.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class local_free_on_block_exit
: private noncopyable
{
public:
explicit local_free_on_block_exit(void* p)
: p_(p)
{
}

~local_free_on_block_exit()
{
::LocalFree(p_);
}

private:
void* p_;
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
#endif 

#endif 
