#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <malloc.h>

#ifndef __INTEL_COMPILER

#include <x86intrin.h>

#endif

#include <omp.h>

#define FREE(x) { if (x) _mm_free(x); x = NULL; }
#define MALLOC(type, len) (type *)_mm_malloc(sizeof(type)*(len), 64)

#include <cstdint>
#include <cassert>
#include <cstdlib>

typedef uint32_t offset_type;

namespace SpMP {


double get_cpu_freq();

template<typename U, typename T>
bool operator<(const std::pair<U, T> &a, const std::pair<U, T> &b) {
if (a.first != b.first) {
return a.first < b.first;
} else {
return a.second < b.second;
}
}


void getLoadBalancedPartition(offset_type *begin, offset_type *end, const offset_type *prefixSum, offset_type n);


bool isPerm(const int *perm, int n);

template<class T>
void copyVector(T *out, const T *in, int len) {
#pragma omp parallel for
for (int i = 0; i < len; ++i) {
out[i] = in[i];
}
}

#define USE_LARGE_PAGE
#ifdef USE_LARGE_PAGE

#include <sys/mman.h>

#define HUGE_PAGE_SIZE (2 * 1024 * 1024)
#define ALIGN_TO_PAGE_SIZE(x) \
(((x) + HUGE_PAGE_SIZE - 1) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE)

#ifndef MAP_HUGETLB
# define MAP_HUGETLB  0x40000
#endif

inline void *malloc_huge_pages(size_t size) {
size_t real_size = ALIGN_TO_PAGE_SIZE(size + HUGE_PAGE_SIZE);
char *ptr = (char *) mmap(NULL, real_size, PROT_READ | PROT_WRITE,
MAP_PRIVATE | MAP_ANONYMOUS |
MAP_POPULATE | MAP_HUGETLB, -1, 0);
if (ptr == MAP_FAILED) {
posix_memalign((void **) &ptr, 4096, real_size);
if (ptr == NULL) return NULL;
real_size = 0;
}
*((size_t *) ptr) = real_size;
return ptr + HUGE_PAGE_SIZE;
}

inline void free_huge_pages(void *ptr) {
if (ptr == NULL) return;
void *real_ptr = (char *) ptr - HUGE_PAGE_SIZE;
size_t real_size = *((size_t *) real_ptr);
assert(real_size % HUGE_PAGE_SIZE == 0);
if (real_size != 0)
munmap(real_ptr, real_size);
else
free(real_ptr);
}

#undef USE_LARGE_PAGE
#endif 

} 
