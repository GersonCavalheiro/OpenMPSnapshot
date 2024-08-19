#include <stdio.h>
#include <omp.h>
#include "ompvv.h"
#define N 1000
int test_zero_length_pointer() {
int compute_array[N];
int *p;
int sum = 0, result = 0, errors = 0;
int i;
for (i = 0; i < N; i++)
compute_array[i] = 0;
p = &compute_array[0];
#pragma omp target data map(tofrom: compute_array)
#pragma omp target
{
for (i = 0; i < N; i++)
p[i] = i;
} 
for (i = 0; i < N; i++)
sum = sum + compute_array[i];
for (i = 0; i < N; i++)
result += i;
OMPVV_TEST_AND_SET_VERBOSE(errors, result != sum);
return errors;
}
int main() {
int errors = 0;
int isOffloading;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_zero_length_pointer());
OMPVV_REPORT_AND_RETURN(errors);
}
