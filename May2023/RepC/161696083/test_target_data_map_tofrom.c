#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int test_map_tofrom() {
OMPVV_INFOMSG("test_map_tofrom");
int sum = 0, sum2 = 0, errors = 0;
int *h_array_h = (int *)malloc(N * sizeof(int));
int h_array_s[N];
for (int i = 0; i < N; ++i) {
h_array_h[i] = 0;
h_array_s[i] = 0;
}
#pragma omp target data map(tofrom: h_array_h[0:N])    map(tofrom : h_array_s[0:N]) 
{
#pragma omp target 
{ 
for (int i = 0; i < N; ++i) {
h_array_h[i] += 1;
h_array_s[i] += 1;
}
} 
} 
for (int i = 0; i < N; ++i) {
sum += h_array_h[i];
sum2 += h_array_s[i];
}
free(h_array_h);
OMPVV_TEST_AND_SET_VERBOSE(errors, (N != sum) || (N != sum2));
return errors;
}
int main() {
int errors = 0;
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_tofrom());
OMPVV_REPORT_AND_RETURN(errors);
}
