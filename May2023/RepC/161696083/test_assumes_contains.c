#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
#pragma omp assumes contains(target, parallel, for)
int errors, i;
int test_assumes_contains() {
int arr[N];
for(i = 0; i < N; i++){
arr[i] = i;
}
#pragma omp target map(tofrom: arr)
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE)
#pragma omp for
for(i = 0; i < N; i++){
arr[i] = arr[i]*2;
}
for(i = 0; i < N; i++){
OMPVV_TEST_AND_SET(errors, arr[i] != i*2);
}
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_assumes_contains() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
