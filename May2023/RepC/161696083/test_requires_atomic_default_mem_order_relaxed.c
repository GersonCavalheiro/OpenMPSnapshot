#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#pragma omp requires atomic_default_mem_order(relaxed)
int test_requires_atomic_relaxed() {
OMPVV_INFOMSG("test_requires_atomic_relaxed");
int x = 0, y = 0;
int errors = 0;
omp_set_dynamic(0);
omp_set_num_threads(2);
#pragma omp parallel 
{
int thrd = omp_get_thread_num();
if (thrd == 0) {
x = 10;
#pragma omp flush
#pragma omp atomic write 
y = 1;
} else {
int tmp = 0;
while (tmp == 0) {
#pragma omp atomic read 
tmp = y;
}
#pragma omp flush
OMPVV_TEST_AND_SET_VERBOSE(errors, x != 10);
}
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_requires_atomic_relaxed());
OMPVV_REPORT_AND_RETURN(errors);
}