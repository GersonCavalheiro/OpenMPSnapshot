#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int errors;
int test_default_firstprivate_taskloop() {
int scalar_var = 6;
int arr[N];
int sum =0;
for (int i=0; i<N; i++){
arr[i] = i;
sum += arr[i];
}
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
{	
#pragma omp single 
{  
#pragma omp taskloop default(firstprivate)
for (int i = 0; i < N; i++) {
arr[i] = i + 3;
scalar_var += 7;
}	
}	
}
int new_sum =0;
int wrong_sum =0;
for (int i = 0; i < N; i++) {
new_sum += arr[i];
wrong_sum += i + 3;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_var != 6);
OMPVV_INFOMSG_IF(scalar_var == 0, "Scalar was not initialized in parallel region & not updated");
OMPVV_INFOMSG_IF(scalar_var >  6, "Scalar was not firstprivate, changes made in parallel affected original copy");
OMPVV_TEST_AND_SET_VERBOSE(errors, sum != new_sum);
OMPVV_INFOMSG_IF(new_sum == 0, "Array was not initialized in parallel region properly");
OMPVV_INFOMSG_IF(new_sum == wrong_sum, "Array was not first private, changes made in parallel affected original copy");
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_default_firstprivate_taskloop() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
