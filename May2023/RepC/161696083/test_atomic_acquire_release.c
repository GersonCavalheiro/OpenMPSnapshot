#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int test_atomic_acquire_release() {
OMPVV_INFOMSG("test_atomic_acquire_release");
int x = 0, y = 0;
int errors = 0;
#pragma omp parallel num_threads(2)
{
int thrd = omp_get_thread_num();
if (thrd == 0) {
x = 10;
#pragma omp atomic write release 
y = 1;
} else {
int tmp = 0;
while (tmp == 0) {
#pragma omp atomic read acquire 
tmp = y;
}
OMPVV_TEST_AND_SET(errors, x != 10);
}
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_acquire_release());
OMPVV_REPORT_AND_RETURN(errors);
}
