#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_allocate_allocator() {
int errors = 0;
int* x;
omp_memspace_handle_t x_memspace = omp_default_mem_space;
omp_alloctrait_t x_traits[1] = {{omp_atk_alignment, 64}};
omp_allocator_handle_t x_alloc = omp_init_allocator(x_memspace, 1, x_traits);
#pragma omp allocate(x) allocator(x_alloc)
x = (int *)omp_alloc(N * sizeof(int), x_alloc);
#pragma omp parallel for simd simdlen(16) aligned(x: 64)
for (int i = 0; i < N; i++) {
x[i] = i;
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != i);
}
omp_free(x, x_alloc);
omp_destroy_allocator(x_alloc);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate_allocator() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}