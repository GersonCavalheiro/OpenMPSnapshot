#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_target_task_depend_mutexinoutset() {
OMPVV_INFOMSG("test_task_mutexinoutset");
int errors = 0;
int a, b, c, d;
#pragma omp target map(from: d)
{
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(out: c)
c = 1; 
#pragma omp task depend(out: a)
a = 2; 
#pragma omp task depend(out: b)
b = 3; 
#pragma omp task depend(in: a) depend(mutexinoutset: c)
c += a; 
#pragma omp task depend(in: b) depend(mutexinoutset: c)
c += b; 
#pragma omp task depend(in: c)
d = c; 
}
}
OMPVV_TEST_AND_SET_VERBOSE(errors, d != 6);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_task_depend_mutexinoutset());
OMPVV_REPORT_AND_RETURN(errors);
}
