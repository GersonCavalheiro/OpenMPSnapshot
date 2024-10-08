#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int test_bitxor() {
unsigned int a[N];
int errors = 0;
int num_threads[N];
srand(1);
for (int x = 0; x < N; ++x) {
a[x] = (unsigned int) rand() / (double) (RAND_MAX / 2);
num_threads[x] = -x;
}
unsigned int b = 0;
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
{
#pragma omp loop reduction(^:b)
for (int x = 0; x < N; ++x) {
b = (b ^ a[x]);
}
#pragma omp for
for (int x = 0; x < N; ++x) {
num_threads[x] = omp_get_num_threads();
}
}
unsigned int host_b = 0;
for (int x = 0; x < N; ++x) {
host_b = (host_b ^ a[x]);
}
for (int x = 1; x < N; ++x) {
OMPVV_WARNING_IF(num_threads[x - 1] != num_threads[x], "Test reported differing numbers of threads.  Validity of testing of reduction clause cannot be guaranteed.");
}
OMPVV_WARNING_IF(num_threads[0] == 1, "Test operated with one thread.  Reduction clause cannot be tested.");
OMPVV_WARNING_IF(num_threads[0] <= 0, "Test reported invalid number of threads.  Validity of testing of reduction clause cannot be guaranteed.");
OMPVV_TEST_AND_SET_VERBOSE(errors, b != host_b);
OMPVV_ERROR_IF(host_b != b, "Bit in loop is %d but expected bit is %d.", b, host_b);
return errors;
}
int main() {
int total_errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_bitxor() != 0);
OMPVV_REPORT_AND_RETURN(total_errors);
}
