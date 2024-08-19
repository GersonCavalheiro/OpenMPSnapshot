#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors, i;
int test_assume_no_parallelism() {
int x;
int arr[N];
#pragma omp assume no_parallelism
{
x = omp_get_thread_num(); 
for(i = 0; i < N; i++){
arr[i] = i + x;
}
}
#pragma omp target parallel for map(tofrom: arr)
for(i = 0; i < N; i++){
arr[i] = arr[i]*2;
}
for(i = 0; i < N; i++){
OMPVV_TEST_AND_SET(errors, arr[i] != (i+x)*2);
}
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_assume_no_parallelism() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
