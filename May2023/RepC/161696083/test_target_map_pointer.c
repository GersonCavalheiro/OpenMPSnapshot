#include <stdio.h>
#include <omp.h>
#include "ompvv.h"
#define N 1000
int main() {
int compute_array[N];
int *p;
int sum = 0, result = 0, errors = 0;
int i;
for (i = 0; i < N; i++)
compute_array[i] = 0;
p = &compute_array[0];
int isOffloading;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "This test is running on host, the value of p[] is not copied over to the device"); 
#pragma omp target data map(tofrom: compute_array) 
#pragma omp target map(to: p[:N]) 
{
for (i = 0; i < N; i++)
p[i] = i;
} 
for (i = 0; i < N; i++)
sum = sum + compute_array[i];    
for (i = 0; i < N; i++)
result += i;
OMPVV_TEST_AND_SET_VERBOSE(errors, result != sum);
OMPVV_REPORT_AND_RETURN(errors);
}
