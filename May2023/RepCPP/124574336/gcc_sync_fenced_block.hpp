
#ifndef BOOST_ASIO_DETAIL_GCC_SYNC_FENCED_BLOCK_HPP
#define BOOST_ASIO_DETAIL_GCC_SYNC_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(__GNUC__) \
&& ((__GNUC__ == 4 && __GNUC_MINOR__ >= 1) || (__GNUC__ > 4)) \
&& !defined(__INTEL_COMPILER) && !defined(__ICL) \
&& !defined(__ICC) && !defined(__ECC) && !defined(__PATHSCALE__)

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class gcc_sync_fenced_block
: private noncopyable
{
public:
enum half_or_full_t { half, full };

explicit gcc_sync_fenced_block(half_or_full_t)
: value_(0)
{
__sync_lock_test_and_set(&value_, 1);
}

~gcc_sync_fenced_block()
{
__sync_lock_release(&value_);
}

private:
int value_;
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
