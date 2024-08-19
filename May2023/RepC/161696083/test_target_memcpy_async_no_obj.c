#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors, i;
int test_target_memcpy_async_no_obj() {
int h, t, i;
errors = 0;
double *mem;
double *mem_dev_cpy;
h = omp_get_initial_device();
t = omp_get_default_device();
OMPVV_TEST_AND_SET_VERBOSE(errors, omp_get_num_devices() < 1 || t < 0);
mem = (double *)malloc( sizeof(double)*N);
mem_dev_cpy = (double *)omp_target_alloc( sizeof(double)*N, t);
OMPVV_TEST_AND_SET_VERBOSE(errors, mem_dev_cpy == NULL);
omp_target_memcpy_async(mem_dev_cpy, mem, sizeof(double)*N,
0,          0,
t,          h,
0,          NULL);
#pragma omp target is_device_ptr(mem_dev_cpy) device(t)
#pragma omp teams distribute parallel for
for(i = 0; i < N; i++){
mem_dev_cpy[i] = i*2; 
}
omp_target_memcpy_async(mem, mem_dev_cpy, sizeof(double)*N,
0,          0,
h,          t,
0,          NULL);
#pragma omp taskwait
for(i = 0; i < N; i++){
OMPVV_TEST_AND_SET(errors, mem[i]!=i*2);
}
free(mem);
omp_target_free(mem_dev_cpy, t);
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_memcpy_async_no_obj() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}