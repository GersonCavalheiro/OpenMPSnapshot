#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors, i;
int test_assume_absent() {
int arr[N];
for(i = 0; i < N; i++){
arr[i] = i;
}
#pragma omp assume absent(teams, masked, scope, simd)
{
#pragma omp target parallel for map(tofrom: arr)
for(i = 0; i < N; i++){
arr[i] = arr[i]*2;
}
}
for(i = 0; i < N; i++){
OMPVV_TEST_AND_SET(errors, arr[i] != i*2);
}
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_assume_absent() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
