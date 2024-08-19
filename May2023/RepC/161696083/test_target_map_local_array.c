#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 10000
int main() {
int compute_array[N];
int sum = 0, errors = 0, result = 0;
int i;
for (i = 0; i < N; i++) 
compute_array[i] = 0;
int isOffloading;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "This test is running on host, array is not allocated on device");
#pragma omp target map(tofrom: compute_array[0:N])
{
for (i = 0; i < N; i++)
compute_array[i] = i;
} 
for (i = 0; i < N; i++){
sum = sum + compute_array[i];    
result += i;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, result != sum);
OMPVV_REPORT_AND_RETURN(errors);
}
