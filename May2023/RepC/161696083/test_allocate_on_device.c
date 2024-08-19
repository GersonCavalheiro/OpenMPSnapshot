#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_allocate_on_device() {
int A[N], errors = 0;
#pragma omp target map(tofrom: errors, A) uses_allocators(omp_default_mem_alloc)
{
int* x;
#pragma omp allocate(x) allocator(omp_default_mem_alloc)
x = (int *) omp_alloc(N*sizeof(int), omp_default_mem_alloc);
#pragma omp parallel for 
for (int i = 0; i < N; i++) {
x[i] = 2*i;
}
for (int i = 0; i < N; i++) {
A[i] = x[i] + i;
}
omp_free(x, omp_default_mem_alloc);
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != 3*i);
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate_on_device() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
