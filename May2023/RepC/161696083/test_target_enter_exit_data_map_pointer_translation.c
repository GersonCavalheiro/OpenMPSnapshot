#include "ompvv.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000
int test_map_same_function() {
OMPVV_INFOMSG("Testing map same function definition")
int sum = 0, sum2 = 0, errors = 0;
int *h_array_h = (int *)malloc(N * sizeof(int));
int h_array_s[N];
int *ptr_h_array_h;
int *ptr_h_array_s;
#pragma omp target enter data map(alloc:h_array_h[0:N]) map(alloc:h_array_s[0:N])
ptr_h_array_h = h_array_h;
ptr_h_array_s = h_array_s;
OMPVV_INFOMSG("map(ptr) specified full-length array section")
#pragma omp target map(ptr_h_array_h[0:N], ptr_h_array_s[0:N])
{
for (int i = 0; i < N; ++i) {
ptr_h_array_h[i] = 1;
ptr_h_array_s[i] = 2;
}
} 
OMPVV_INFOMSG("map(ptr) specified zero-lenght array section")
#pragma omp target map(ptr_h_array_h[:0], ptr_h_array_s[:0])
{
for (int i = 0; i < N; ++i) {
ptr_h_array_h[i] += 1;
ptr_h_array_s[i] += 2;
}
} 
OMPVV_INFOMSG("no map(ptr) Specified")
#pragma omp target 
{
for (int i = 0; i < N; ++i) {
ptr_h_array_h[i] += 1;
ptr_h_array_s[i] += 2;
}
} 
#pragma omp target exit data map(from:h_array_h[0:N]) map(from:h_array_s[0:N])
for (int i = 0; i < N; ++i) {
sum += h_array_h[i];
sum2 += h_array_s[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (3*N != sum) || (6*N != sum2));
free(h_array_h);
return errors;
}
void helper_function(int *ptr_h_array_h, int *ptr_h_array_s) {
OMPVV_INFOMSG("map(ptr) specified full-lenght array section")
#pragma omp target map(ptr_h_array_h[0:N], ptr_h_array_s[0:N])
{
for (int i = 0; i < N; ++i) {
ptr_h_array_h[i] = 1;
ptr_h_array_s[i] = 2;
}
} 
OMPVV_INFOMSG("map(ptr) specified zero-lenght array section")
#pragma omp target map(ptr_h_array_h[:0], ptr_h_array_s[:0])
{
for (int i = 0; i < N; ++i) {
ptr_h_array_h[i] += 1;
ptr_h_array_s[i] += 2;
}
} 
OMPVV_INFOMSG("no map(ptr) Specified")
#pragma omp target 
{
for (int i = 0; i < N; ++i) {
ptr_h_array_h[i] += 1;
ptr_h_array_s[i] += 2;
}
} 
}
int test_map_different_function() {
OMPVV_INFOMSG("Testing map different function definition")
int sum = 0, sum2 = 0, errors = 0;
int *h_array_h = (int *)malloc(N * sizeof(int));
int h_array_s[N];
#pragma omp target enter data map(alloc: h_array_h[0:N]) map(alloc: h_array_s[0:N])
helper_function(h_array_h, h_array_s);
#pragma omp target exit data map(from: h_array_h[0:N]) map(from: h_array_s[0:N])
for (int i = 0; i < N; ++i) {
sum += h_array_h[i];
sum2 += h_array_s[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (3*N != sum) || (6*N != sum2));
free(h_array_h);
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_same_function());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_different_function());
OMPVV_REPORT_AND_RETURN(errors);
}
