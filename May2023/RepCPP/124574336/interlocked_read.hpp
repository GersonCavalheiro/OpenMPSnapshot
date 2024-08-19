#ifndef BOOST_THREAD_DETAIL_INTERLOCKED_READ_WIN32_HPP
#define BOOST_THREAD_DETAIL_INTERLOCKED_READ_WIN32_HPP


#include <boost/detail/interlocked.hpp>
#include <boost/thread/detail/config.hpp>

#include <boost/config/abi_prefix.hpp>

#if defined(__INTEL_COMPILER)
#define BOOST_THREAD_DETAIL_COMPILER_BARRIER() __memory_barrier()
#elif defined(__clang__)
#define BOOST_THREAD_DETAIL_COMPILER_BARRIER() __atomic_signal_fence(__ATOMIC_SEQ_CST)
#elif defined(_MSC_VER) && !defined(_WIN32_WCE)
extern "C" void _ReadWriteBarrier(void);
#pragma intrinsic(_ReadWriteBarrier)
#define BOOST_THREAD_DETAIL_COMPILER_BARRIER() _ReadWriteBarrier()
#endif

#ifndef BOOST_THREAD_DETAIL_COMPILER_BARRIER
#define BOOST_THREAD_DETAIL_COMPILER_BARRIER()
#endif

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))


namespace boost
{
namespace detail
{
inline long interlocked_read_acquire(long volatile* x) BOOST_NOEXCEPT
{
long const res=*x;
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
return res;
}
inline void* interlocked_read_acquire(void* volatile* x) BOOST_NOEXCEPT
{
void* const res=*x;
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
return res;
}

inline void interlocked_write_release(long volatile* x,long value) BOOST_NOEXCEPT
{
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
*x=value;
}
inline void interlocked_write_release(void* volatile* x,void* value) BOOST_NOEXCEPT
{
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
*x=value;
}
}
}

#elif defined(_MSC_VER) && _MSC_VER >= 1700 && (defined(_M_ARM) || defined(_M_ARM64))

#include <intrin.h>

namespace boost
{
namespace detail
{
inline long interlocked_read_acquire(long volatile* x) BOOST_NOEXCEPT
{
long const res=__iso_volatile_load32((const volatile __int32*)x);
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
__dmb(0xB); 
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
return res;
}
inline void* interlocked_read_acquire(void* volatile* x) BOOST_NOEXCEPT
{
void* const res=
#if defined(_M_ARM64)
(void*)__iso_volatile_load64((const volatile __int64*)x);
#else
(void*)__iso_volatile_load32((const volatile __int32*)x);
#endif
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
__dmb(0xB); 
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
return res;
}

inline void interlocked_write_release(long volatile* x,long value) BOOST_NOEXCEPT
{
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
__dmb(0xB); 
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
__iso_volatile_store32((volatile __int32*)x, (__int32)value);
}
inline void interlocked_write_release(void* volatile* x,void* value) BOOST_NOEXCEPT
{
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
__dmb(0xB); 
BOOST_THREAD_DETAIL_COMPILER_BARRIER();
#if defined(_M_ARM64)
__iso_volatile_store64((volatile __int64*)x, (__int64)value);
#else
__iso_volatile_store32((volatile __int32*)x, (__int32)value);
#endif
}
}
}

#elif defined(__GNUC__) && (((__GNUC__ * 100 + __GNUC_MINOR__) >= 407) || (defined(__clang__) && (__clang_major__ * 100 + __clang_minor__) >= 302))

namespace boost
{
namespace detail
{
inline long interlocked_read_acquire(long volatile* x) BOOST_NOEXCEPT
{
return __atomic_load_n((long*)x, __ATOMIC_ACQUIRE);
}
inline void* interlocked_read_acquire(void* volatile* x) BOOST_NOEXCEPT
{
return __atomic_load_n((void**)x, __ATOMIC_ACQUIRE);
}

inline void interlocked_write_release(long volatile* x,long value) BOOST_NOEXCEPT
{
__atomic_store_n((long*)x, value, __ATOMIC_RELEASE);
}
inline void interlocked_write_release(void* volatile* x,void* value) BOOST_NOEXCEPT
{
__atomic_store_n((void**)x, value, __ATOMIC_RELEASE);
}
}
}

#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))

namespace boost
{
namespace detail
{
inline long interlocked_read_acquire(long volatile* x) BOOST_NOEXCEPT
{
long res;
__asm__ __volatile__ ("movl %1, %0" : "=r" (res) : "m" (*x) : "memory");
return res;
}
inline void* interlocked_read_acquire(void* volatile* x) BOOST_NOEXCEPT
{
void* res;
#if defined(__x86_64__)
__asm__ __volatile__ ("movq %1, %0" : "=r" (res) : "m" (*x) : "memory");
#else
__asm__ __volatile__ ("movl %1, %0" : "=r" (res) : "m" (*x) : "memory");
#endif
return res;
}

inline void interlocked_write_release(long volatile* x,long value) BOOST_NOEXCEPT
{
__asm__ __volatile__ ("movl %1, %0" : "=m" (*x) : "r" (value) : "memory");
}
inline void interlocked_write_release(void* volatile* x,void* value) BOOST_NOEXCEPT
{
#if defined(__x86_64__)
__asm__ __volatile__ ("movq %1, %0" : "=m" (*x) : "r" (value) : "memory");
#else
__asm__ __volatile__ ("movl %1, %0" : "=m" (*x) : "r" (value) : "memory");
#endif
}
}
}

#else

namespace boost
{
namespace detail
{
inline long interlocked_read_acquire(long volatile* x) BOOST_NOEXCEPT
{
return BOOST_INTERLOCKED_COMPARE_EXCHANGE((long*)x,0,0);
}
inline void* interlocked_read_acquire(void* volatile* x) BOOST_NOEXCEPT
{
return BOOST_INTERLOCKED_COMPARE_EXCHANGE_POINTER((void**)x,0,0);
}
inline void interlocked_write_release(long volatile* x,long value) BOOST_NOEXCEPT
{
BOOST_INTERLOCKED_EXCHANGE((long*)x,value);
}
inline void interlocked_write_release(void* volatile* x,void* value) BOOST_NOEXCEPT
{
BOOST_INTERLOCKED_EXCHANGE_POINTER((void**)x,value);
}
}
}

#endif

#include <boost/config/abi_suffix.hpp>

#endif
