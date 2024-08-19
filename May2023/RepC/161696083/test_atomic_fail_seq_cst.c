#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_atomic_fail_seq_cst() {
OMPVV_INFOMSG("test_atomic_fail_seq_cst");
int x = 0, y = 0;
int errors = 0;
#pragma omp parallel num_threads(2)
{
int thrd = omp_get_thread_num();
if (thrd == 0) {
y = 1;
#pragma omp atomic write release 
x = 10;
} else {
int tmp = 0;
while (y != 5) {
#pragma omp atomic compare acquire fail(seq_cst)
if(y == 1){
y = 5;
}
}
OMPVV_TEST_AND_SET(errors, x != 10);
OMPVV_TEST_AND_SET(errors, y != 5);
}
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_fail_seq_cst());
OMPVV_REPORT_AND_RETURN(errors);
}
