#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
int test_atomic_with_used_enum_value() {
OMPVV_INFOMSG("test_atomic_with_used_enum_value");
int errors = 0, x = 0, num_threads = -1;
#pragma omp target map(tofrom: num_threads,x)  
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE) default(shared)
{
if (omp_get_thread_num() == 0) {
num_threads = omp_get_num_threads();
}
#pragma omp atomic hint(0X4) 
x++;
}
OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of threads (less than zero)");
OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");
OMPVV_TEST_AND_SET_VERBOSE(errors, x != num_threads);
return errors;
}
int test_atomic_with_unused_enum_value() {
OMPVV_INFOMSG("test_atomic_with_unused_enum_value");
int errors = 0, x = 0, num_threads = -1;
#pragma omp target map(tofrom: num_threads,x)  
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE) default(shared)
{
if (omp_get_thread_num() == 0) {
num_threads = omp_get_num_threads();
}
#pragma omp atomic hint(0X1024) 
x++;
}
OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of threads (less than zero)");
OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");
OMPVV_TEST_AND_SET_VERBOSE(errors, x != num_threads);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_with_used_enum_value());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_with_unused_enum_value());
OMPVV_REPORT_AND_RETURN(errors);
}
