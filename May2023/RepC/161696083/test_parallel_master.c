#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_parallel_master() {
OMPVV_INFOMSG("test_parallel_master");
int errors = 0;
int num_threads = -1;
int x[N];
int y[N];
int z[N];
for (int i = 0; i < N; i++) {
x[i] = 1;
y[i] = i + 1;
z[i] = 2*(i + 1);
}
#pragma omp parallel master num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z, num_threads)
{
#pragma omp taskloop
for (int i = 0; i < N; i++) {
x[i] += y[i]*z[i];
}
num_threads = omp_get_num_threads();
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != 1 + (y[i]*z[i]));
}
OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of parallel master with taskloop can't be guaranteed.");
OMPVV_ERROR_IF(num_threads < 1, "Test returned an invalid number of threads.");
OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_parallel_master());
OMPVV_REPORT_AND_RETURN(errors);
}
