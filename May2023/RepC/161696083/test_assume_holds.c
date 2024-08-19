#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
int errors, i, N;
int test_assume_holds() {
N = 1024;
int arr[N];
#pragma omp assume holds(N == 1024)
{
for(i = 0; i < N; i++){
arr[i] = i;
}
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
OMPVV_TEST_AND_SET_VERBOSE(errors, test_assume_holds() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
