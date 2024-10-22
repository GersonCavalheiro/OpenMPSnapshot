#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors;
int test_default_firstprivate_task(){
int test_num = 1;
int test_arr[N];
int sum = 0;
for (int i = 0; i<N; i++){
test_arr[i] = i;
sum += i;
}
#pragma omp task default(firstprivate)
{
test_num += 1;
for (int i=0; i<N; i++){
test_arr[i] = 1;
}
}
int new_sum = 0;
int wrong_sum = 0;
for (int i = 0; i<N; i++){
new_sum += test_arr[i];
wrong_sum += 1;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, test_num != 1);
OMPVV_INFOMSG_IF(test_num == 2, "Scalar was not firstprivate, changes made in task affected original copy");
OMPVV_TEST_AND_SET_VERBOSE(errors, sum != new_sum);
OMPVV_INFOMSG_IF(new_sum == wrong_sum, "Array was not first private, changes made in task affected original copy");
return errors;
}
int main(){
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_default_firstprivate_task() != 0);
OMPVV_REPORT_AND_RETURN(errors);	
}
