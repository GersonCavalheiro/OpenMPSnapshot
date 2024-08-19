#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 1000
int main() {
int compute_array[N];
int sum = 0, result = 0, errors = 0;
int i;
OMPVV_TEST_OFFLOADING;
for (i=0; i<N; i++) 
compute_array[i] = 10;
#pragma omp target map(compute_array)
{
for (i = 0; i < N; i++)
compute_array[i] += i;
} 
for (i = 0; i < N; i++)
sum = sum + compute_array[i];
for (i = 0; i < N; i++)
result += 10 + i;
OMPVV_TEST_AND_SET_VERBOSE(errors, result != sum);
OMPVV_REPORT_AND_RETURN(errors)
}
