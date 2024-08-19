#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int test_map_to_from() {
OMPVV_INFOMSG("test_map_to_from");
int sum = 0, errors = 0;
int *h_array_h = (int *)malloc(N * sizeof(int));
int *h_array2_h = (int *)malloc(N * sizeof(int));
for (int i = 0; i < N; ++i) {
h_array_h[i] = 1;
h_array2_h[i] = 0;
}
#pragma omp target data map(to: h_array_h[0:N]) map(from: h_array2_h[0:N])  
{
#pragma omp target 
{
for (int i = 0; i < N; ++i)
h_array2_h[i] = h_array_h[i];
} 
} 
for (int i = 0; i < N; ++i)
sum += h_array2_h[i];
free(h_array_h);
free(h_array2_h);
OMPVV_TEST_AND_SET_VERBOSE(errors, ((N - sum) != 0));
return errors;
}
int main() {
int errors = 0;
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_to_from());
OMPVV_REPORT_AND_RETURN(errors);
}
