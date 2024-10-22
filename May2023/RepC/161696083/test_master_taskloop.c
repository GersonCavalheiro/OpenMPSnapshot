#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_master_taskloop() {
OMPVV_INFOMSG("test_master_taskloop");
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
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST) shared(x, y, z, num_threads)
{
#pragma omp master taskloop
for (int i = 0; i < N; i++) {
x[i] += y[i]*z[i];
}
if (omp_get_thread_num() == 0) {
num_threads = omp_get_num_threads();
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != 1 + (y[i]*z[i]));
}
OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so parallelism of taskloop can't be guaranteed.");
OMPVV_ERROR_IF(num_threads < 1, "Test returned an invalid number of threads.");
OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_master_taskloop());
OMPVV_REPORT_AND_RETURN(errors);
}
