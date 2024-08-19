
#ifndef ASIO_DETAIL_GCC_X86_FENCED_BLOCK_HPP
#define ASIO_DETAIL_GCC_X86_FENCED_BLOCK_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"

#if defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))

#include "asio/detail/noncopyable.hpp"

#include "asio/detail/push_options.hpp"

namespace asio {
namespace detail {

class gcc_x86_fenced_block
: private noncopyable
{
public:
enum half_t { half };
enum full_t { full };

explicit gcc_x86_fenced_block(half_t)
{
}

explicit gcc_x86_fenced_block(full_t)
{
lbarrier();
}

~gcc_x86_fenced_block()
{
sbarrier();
}

private:
static int barrier()
{
int r = 0, m = 1;
__asm__ __volatile__ (
"xchgl %0, %1" :
"=r"(r), "=m"(m) :
"0"(1), "m"(m) :
"memory", "cc");
return r;
}

static void lbarrier()
{
#if defined(__SSE2__)
# if (__GNUC__ >= 4) && !defined(__INTEL_COMPILER) && !defined(__ICL)
__builtin_ia32_lfence();
# else 
__asm__ __volatile__ ("lfence" ::: "memory");
# endif 
#else 
barrier();
#endif 
}

static void sbarrier()
{
#if defined(__SSE2__)
# if (__GNUC__ >= 4) && !defined(__INTEL_COMPILER) && !defined(__ICL)
__builtin_ia32_sfence();
# else 
__asm__ __volatile__ ("sfence" ::: "memory");
# endif 
#else 
barrier();
#endif 
}
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 

#endif 
