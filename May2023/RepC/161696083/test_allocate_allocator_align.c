#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_allocate_allocator_align() {
int errors = 0;
int x[N];
#pragma omp allocate(x) allocator(omp_default_mem_alloc) align(64)
#pragma omp target map(from:x[:N])
{	
#pragma omp parallel for simd simdlen(16) aligned(x: 64)
for (int i = 0; i < N; i++) {
x[i] = i;
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != i);
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate_allocator_align() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
