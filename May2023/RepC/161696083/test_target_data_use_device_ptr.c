#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int main() {
int errors = 0, map_dummy;
int *array_device = NULL;
int *array_host = NULL;
array_device = (int *)malloc(N*sizeof(int));
array_host = (int *)malloc(N*sizeof(int));
for (int i = 0; i < N; ++i)
array_host[i] = i;
OMPVV_TEST_OFFLOADING;
#pragma omp target data map(tofrom: array_device[0:N])
{
#pragma omp target data map(map_dummy) use_device_ptr(array_device)
{
#pragma omp target is_device_ptr(array_device) map(tofrom: array_host[0:N])
{
for (int i = 0; i < N; ++i) {
array_device[i] = i;
array_host[i] += array_device[i];
} 
} 
} 
} 
for (int i = 0; i < N; ++i) {
OMPVV_TEST_AND_SET(errors, array_host[i] != 2*i);
}
free(array_device);
free(array_host);
OMPVV_REPORT_AND_RETURN(errors);
}
