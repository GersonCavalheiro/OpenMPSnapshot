#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_pointer_swap() {
int* a = (int *) malloc(N * sizeof(int));
int* b = (int *) malloc(N * sizeof(int));
int* temp;
int is_offloading;
int errors = 0;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
for (int x = 0; x < N; ++x) {
a[x] = x;
b[x] = 0;
}
#pragma omp target data map(tofrom: a[0:N]) map(to: b[0:N])
{
#pragma omp target map(alloc: a[0:N], b[0:N])
for (int x = 0; x < N; ++x) {
b[x] = a[x]*2;
}
temp = a;
a = b;
b = temp;
}
for (int x = 0; x < N; ++x) {
OMPVV_TEST_AND_SET(errors, b[x] != x);
if (is_offloading) {
OMPVV_TEST_AND_SET(errors, a[x] != 0);
} else {
OMPVV_TEST_AND_SET(errors, a[x] != 2*x);
}
}
free(a);
free(b);
return errors;
}
int test_pointer_swap_with_update() {
int* a = (int *) malloc(N * sizeof(int));
int* b = (int *) malloc(N * sizeof(int));
int* temp;
int errors = 0;
for (int x = 0; x < N; ++x) {
a[x] = x;
b[x] = 0;
}
#pragma omp target data map(tofrom: a[0:N]) map(to: b[0:N])
{
#pragma omp target map(alloc: a[0:N], b[0:N])
for (int x = 0; x < N; ++x) {
b[x] = a[x]*2;
}
temp = a;
a = b;
b = temp;
#pragma omp target update from(a[0:N])
}
for (int x = 0; x < N; ++x) {
OMPVV_TEST_AND_SET(errors, b[x] != x);
OMPVV_TEST_AND_SET(errors, a[x] != 2*x);
}
free(a);
free(b);
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_pointer_swap());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_pointer_swap_with_update());
OMPVV_REPORT_AND_RETURN(errors);
}
