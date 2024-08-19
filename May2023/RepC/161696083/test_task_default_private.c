#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 1024
int test_task_default_private(){
int errors = 0;
int test_num= 1;
int sum = 0;
#pragma omp task shared(sum) default(private)
{
#pragma omp target map(tofrom: sum, test_num)
{
int test_num = 2;
sum += test_num;
}
}	
OMPVV_TEST_AND_SET_VERBOSE(errors, sum != 2);
OMPVV_INFOMSG_IF(sum == 1, "Did not use variable delcared in task region");
return errors;
}
int main(){
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_default_private() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
