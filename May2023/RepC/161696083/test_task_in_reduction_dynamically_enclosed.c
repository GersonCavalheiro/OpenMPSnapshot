#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int sum = 0;
void task_container(int i) {
#pragma omp task in_reduction(+:sum)
sum += 1 + i;
#pragma omp taskwait
#pragma omp task in_reduction(+:sum)
sum += 1 + i*2;
#pragma omp taskwait
}
int test_task_in_reduction_dynamically_enclosed() {
OMPVV_INFOMSG("test_task_in_reduction_dynamically_enclosed");
int errors = 0;
int expect = 2;
#pragma omp taskgroup task_reduction(+:sum)
task_container(0);
#pragma omp taskloop reduction(+:sum)
for (int i = 0; i < N; i++) {
task_container(i);
sum += i;
}
for (int i = 0; i < N; i++) {
expect += 2 + 4*i;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, sum != expect);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_in_reduction_dynamically_enclosed());
OMPVV_REPORT_AND_RETURN(errors);
}
