

#pragma once

#ifndef NDEBUG

#include <cassert> 

#define invariant(expr) assert(expr)

#else 

#ifndef __has_builtin      
#define __has_builtin(x) 0 
#endif

#if __has_builtin(__builtin_assume)

#define invariant(expr) __builtin_assume(expr)

#else 

namespace rawspeed {

__attribute__((always_inline)) constexpr inline void invariant(bool precond) {
if (!precond)
__builtin_unreachable();
}

} 

#endif 

#endif 
