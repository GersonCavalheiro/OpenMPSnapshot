
#ifndef BOOST_ASIO_DETAIL_GCC_HPPA_FENCED_BLOCK_HPP
#define BOOST_ASIO_DETAIL_GCC_HPPA_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(__GNUC__) && (defined(__hppa) || defined(__hppa__))

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class gcc_hppa_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit gcc_hppa_fenced_block(half_t)
{
}

explicit gcc_hppa_fenced_block(full_t)
{
barrier();
}

~gcc_hppa_fenced_block()
{
barrier();
}

private:
static void barrier()
{
__asm__ __volatile__ ("" : : : "memory");
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
