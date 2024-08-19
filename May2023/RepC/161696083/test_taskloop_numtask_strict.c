#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors;
int test_taskloop_strict_numtasks() {
int arr[N];
int sum = 0;
int parallel_sum = 0; 
for (int i=0; i<N; i++){
arr[i] = 1;
sum += arr[i];
}
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
#pragma omp single
#pragma omp taskloop num_tasks(strict: 100) reduction(+: parallel_sum)
for (int i = 0; i < N; i++) {
parallel_sum += arr[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, parallel_sum != sum);
OMPVV_INFOMSG_IF(sum == 0, "Array was not initialized.");
OMPVV_INFOMSG_IF(parallel_sum == 0, "Data sharing of parallel_sum was wrong.");
OMPVV_INFOMSG_IF(parallel_sum == sum, "Test passed.");
OMPVV_INFOMSG_IF(parallel_sum != sum, "Test did not pass.");
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_taskloop_strict_numtasks() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
