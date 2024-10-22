#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define SIZE 1024
int main() {
OMPVV_TEST_OFFLOADING;
int a[SIZE];
int b[SIZE];
int c[SIZE];
int d[SIZE];
int privatized;
int errors = 0;
int num_threads = -1;
for (int x = 0; x < SIZE; ++x) {
a[x] = 1;
b[x] = x;
c[x] = 2*x;
d[x] = 0;
}
#pragma omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b, c, d, num_threads)
{
#pragma omp loop private(privatized)
for (int x = 0; x < SIZE; ++x) {
privatized = 0;
for (int y = 0; y < a[x] + b[x]; ++y) {
privatized++;
}
d[x] = c[x] * privatized;
}
if (omp_get_thread_num() == 0) {
num_threads = omp_get_num_threads();
}
}
for (int x = 0; x < SIZE; ++x) {
OMPVV_TEST_AND_SET_VERBOSE(errors, d[x] != (1 + x)*2*x);
if (d[x] != (1 + x)*2*x) {
break;
}
}
OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread. Results of private test are inconclusive.");
OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads < 1);
OMPVV_REPORT_AND_RETURN(errors);
}
