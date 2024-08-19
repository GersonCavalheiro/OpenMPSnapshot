#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#include <unistd.h>
#define N 1024
int test_task_nowait(){
int errors = 0;
int test_scalar = 1;
int test_arr[N];
int sum = 0;
for (int i =0; i<N; i++){
test_arr[i] = 1;
sum += i;
}
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(inout: test_scalar) shared(test_scalar)
{
usleep(10);
test_scalar += 1;
}
#pragma omp taskwait nowait depend(inout: test_scalar) depend(out: test_arr)
#pragma omp task depend(inout : test_arr) shared(test_arr)
{
for (int i=0; i<N; i++){
test_arr[i] += 1;
}
}
}
int new_sum = 0;
for (int i = 0; i < N; i++){
new_sum += test_arr[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, test_scalar != 2);
OMPVV_INFOMSG_IF(test_scalar == 1, "Scalar task region failed");
OMPVV_TEST_AND_SET_VERBOSE(errors, new_sum != 2048);
OMPVV_INFOMSG_IF(sum == new_sum, "Array taskwait region failed");
return errors;
}
int main(){
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_nowait() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
