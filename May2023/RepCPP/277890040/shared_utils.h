

#ifndef __TBB_shared_utils_H
#define __TBB_shared_utils_H

#include <stddef.h>  
#if _MSC_VER
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#if !UINTPTR_MAX
#define UINTPTR_MAX SIZE_MAX
#endif
#else 
#include <stdint.h>
#endif


template<typename T>
static inline T alignDown(T arg, uintptr_t alignment) {
return T( (uintptr_t)arg                & ~(alignment-1));
}
template<typename T>
static inline T alignUp  (T arg, uintptr_t alignment) {
return T(((uintptr_t)arg+(alignment-1)) & ~(alignment-1));
}
template<typename T> 
static inline T alignUpGeneric(T arg, uintptr_t alignment) {
if (size_t rem = arg % alignment) {
arg += alignment - rem;
}
return arg;
}

template<typename T, size_t N> 
inline size_t arrayLength(const T(&)[N]) {
return N;
}


template <size_t NUM>
struct Log2 { static const int value = 1 + Log2<(NUM >> 1)>::value; };
template <>
struct Log2<1> { static const int value = 0; };

#if defined(min)
#undef min
#endif

template<typename T>
T min ( const T& val1, const T& val2 ) {
return val1 < val2 ? val1 : val2;
}



#include <stdio.h>

#if defined(_MSC_VER) && (_MSC_VER<1900) && !defined(__INTEL_COMPILER)
#pragma warning(push)
#pragma warning(disable:4510 4512 4610)
#endif

#if __SUNPRO_CC
#pragma error_messages (off, refmemnoconstr)
#endif

struct parseFileItem {
const char* format;
unsigned long long& value;
};

#if defined(_MSC_VER) && (_MSC_VER<1900) && !defined(__INTEL_COMPILER)
#pragma warning(pop)
#endif

#if __SUNPRO_CC
#pragma error_messages (on, refmemnoconstr)
#endif

template <int BUF_LINE_SIZE, int N>
void parseFile(const char* file, const parseFileItem (&items)[N]) {
int found[N] = { 0 };
int numFound = 0;
char buf[BUF_LINE_SIZE];

if (FILE *f = fopen(file, "r")) {
while (numFound < N && fgets(buf, BUF_LINE_SIZE, f)) {
for (int i = 0; i < N; ++i) {
if (!found[i] && 1 == sscanf(buf, items[i].format, &items[i].value)) {
++numFound;
found[i] = 1;
}
}
}
fclose(f);
}
}

namespace rml {
namespace internal {


#if __powerpc64__ || __ppc64__ || __bgp__
const uint32_t estimatedCacheLineSize = 128;
#else
const uint32_t estimatedCacheLineSize =  64;
#endif

} 
} 

#endif 

