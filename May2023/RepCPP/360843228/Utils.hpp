

#ifndef UTILS_HPP
#define UTILS_HPP

#include <config.h>

#include <cstddef>
#include <cstdint>

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif


#ifdef NDEBUG
#define UNUSED_VARIABLE(_var) (void)(_var)
#else
#define UNUSED_VARIABLE(_var)
#endif

#ifndef CACHELINE_SIZE
#define CACHELINE_SIZE 64
#endif

#ifndef MAX_SYSTEM_CPUS
#define MAX_SYSTEM_CPUS 50
#endif

#define MASK_BITS(a)    (sizeof(a)*8)
#define MASK_SET(a,b)   ((a) |= (1ULL<<(b)))
#define MASK_CLEAR(a,b) ((a) &= ~(1ULL<<(b)))
#define MASK_ISSET(a,b) ((a) & (1ULL<<(b)))
#define MASK_RESET(a)   ((a) = (0ULL))
#define MASK_COUNT(a)   (__builtin_popcountll(a))

namespace tagaspi {
namespace util {
typedef size_t mask_t;

template<class T, size_t Size = CACHELINE_SIZE>
class Padded : public T {
using T::T;

constexpr static size_t roundup(size_t const x, size_t const y)
{
return (((x + (y - 1)) / y) * y);
}

uint8_t padding[roundup(sizeof(T), Size)-sizeof(T)];

public:
inline T *ptr_to_basetype()
{
return (T *) this;
}
};

static inline void spinWait()
{
#if defined(__powerpc__) || defined(__powerpc64__) || defined(__PPC__) || defined(__PPC64__) || defined(_ARCH_PPC) || defined(_ARCH_PPC64)
#define HMT_low()       asm volatile("or 1,1,1       # low priority")
#define HMT_medium()    asm volatile("or 2,2,2       # medium priority")
#define HMT_barrier()   asm volatile("" : : : "memory")
HMT_low(); HMT_medium(); HMT_barrier();
#elif defined(__arm__) || defined(__aarch64__)
__asm__ __volatile__ ("yield");
#elif defined(__SSE2__)
_mm_pause();
#elif defined(__i386__) || defined(__x86_64__)
asm volatile("pause" ::: "memory");
#else
#pragma message ("No 'pause' instruction/intrisic found for this architecture ")
#endif
}
} 
} 

#endif 
