
#pragma once

#include <atomic>
#include "intrinsics.h"

namespace embree
{

#if defined(__INTEL_COMPILER)
#elif defined(__GNUC__) || defined(__clang__)
#  define __memory_barrier() asm volatile("" ::: "memory")
#elif  defined(_MSC_VER)
#  define __memory_barrier() _ReadWriteBarrier()
#endif

template <typename T>
struct atomic : public std::atomic<T>
{
atomic () {}

atomic (const T& a)
: std::atomic<T>(a) {}

atomic (const atomic<T>& a) {
this->store(a.load());
}

atomic& operator=(const atomic<T>& other) {
this->store(other.load());
return *this;
}
};

template<typename T>
__forceinline void atomic_min(std::atomic<T>& aref, const T& bref)
{
const T b = bref.load();
while (true) {
T a = aref.load();
if (a <= b) break;
if (aref.compare_exchange_strong(a,b)) break;
}
}

template<typename T>
__forceinline void atomic_max(std::atomic<T>& aref, const T& bref)
{
const T b = bref.load();
while (true) {
T a = aref.load();
if (a >= b) break;
if (aref.compare_exchange_strong(a,b)) break;
}
}
}
