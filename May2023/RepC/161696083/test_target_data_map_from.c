#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int test_map_from() {
OMPVV_INFOMSG("test_map_from");
int sum = 0, sum2 = 0, errors = 0;
int *h_array_h = (int *)malloc(N * sizeof(int));
int h_array_s[N];
#pragma omp target data map(from: h_array_h[0:N])  map(from: h_array_s[0:N])
{
#pragma omp target
{
for (int i = 0; i < N; ++i) {
h_array_h[i] = 1;
h_array_s[i] = 2;
}
} 
} 
for (int i = 0; i < N; ++i) {
sum += h_array_h[i];
sum2 += h_array_s[i];
}
free(h_array_h);
OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (2*N != sum2));
return errors;
}
int main() {
int errors = 0;
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_from());
OMPVV_REPORT_AND_RETURN(errors);
}
