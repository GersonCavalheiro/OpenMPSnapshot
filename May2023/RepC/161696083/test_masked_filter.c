#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_masked_filter() {
int errors = 0;
int ct = 0;
int total = 10;
int threads = OMPVV_NUM_THREADS_HOST;
#pragma omp parallel num_threads(threads)
while(1){
int tot;
#pragma omp atomic read
tot = total;
if (tot <= 0)
break;
#pragma omp masked filter(3)
{
OMPVV_TEST_AND_SET_VERBOSE(errors, omp_get_thread_num() != 3); 
ct++;
#pragma omp atomic
total = total-1;
}
}
OMPVV_TEST_AND_SET_VERBOSE(errors, ct != 10);
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_masked_filter() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
