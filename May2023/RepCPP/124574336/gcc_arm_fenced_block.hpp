
#ifndef BOOST_ASIO_DETAIL_GCC_ARM_FENCED_BLOCK_HPP
#define BOOST_ASIO_DETAIL_GCC_ARM_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>

#if defined(__GNUC__) && defined(__arm__)

#include <boost/asio/detail/noncopyable.hpp>

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace detail {

class gcc_arm_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit gcc_arm_fenced_block(half_t)
{
}

explicit gcc_arm_fenced_block(full_t)
{
barrier();
}

~gcc_arm_fenced_block()
{
barrier();
}

private:
static void barrier()
{
#if defined(__ARM_ARCH_4__) \
|| defined(__ARM_ARCH_4T__) \
|| defined(__ARM_ARCH_5__) \
|| defined(__ARM_ARCH_5E__) \
|| defined(__ARM_ARCH_5T__) \
|| defined(__ARM_ARCH_5TE__) \
|| defined(__ARM_ARCH_5TEJ__) \
|| defined(__ARM_ARCH_6__) \
|| defined(__ARM_ARCH_6J__) \
|| defined(__ARM_ARCH_6K__) \
|| defined(__ARM_ARCH_6Z__) \
|| defined(__ARM_ARCH_6ZK__) \
|| defined(__ARM_ARCH_6T2__)
# if defined(__thumb__)
__asm__ __volatile__ ("" : : : "memory");
# else 
int a = 0, b = 0;
__asm__ __volatile__ ("swp %0, %1, [%2]"
: "=&r"(a) : "r"(1), "r"(&b) : "memory", "cc");
# endif 
#else
__asm__ __volatile__ ("dmb" : : : "memory");
#endif
}
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 

#endif 
