#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
int flush_default();
int flush_seq_cst();
int errors = 0;
int flush_default_result = 0;
int flush_seq_cst_result = 0;
int flush_default() { 
int x = 0, y = 0;
omp_set_dynamic(0);   
#pragma omp parallel num_threads(2)
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
flush_default_result = x;
}
}
return errors;
}
int flush_seq_cst() { 
int x = 0, y = 0;
omp_set_dynamic(0);   
#pragma omp parallel num_threads(2)
{
int thrd = omp_get_thread_num();
if (thrd == 0) {
x = 10;
#pragma omp flush seq_cst
#pragma omp atomic write
y = 1;
} else {
int tmp = 0;
while (tmp == 0) {
#pragma omp atomic read
tmp = y;
}
#pragma omp flush seq_cst
OMPVV_TEST_AND_SET_VERBOSE(errors, x != 10);
flush_seq_cst_result = x;
}
}
return errors;
}
int main() {
errors = 0;
flush_default_result = 0;
flush_seq_cst_result = 0;
OMPVV_TEST_OFFLOADING;
flush_default(); 
OMPVV_TEST_AND_SET_VERBOSE(errors, flush_seq_cst() != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, flush_default_result != flush_seq_cst_result);
OMPVV_ERROR_IF(flush_default_result != flush_seq_cst_result, "Error: Flush with seq_cst clause not the same flush with no specified clause");
OMPVV_REPORT_AND_RETURN(errors);
}
