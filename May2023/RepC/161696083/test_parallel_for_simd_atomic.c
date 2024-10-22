#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_parallel_for_simd_atomic() {
OMPVV_INFOMSG("test_parallel_for_simd_atomic");
int errors = 0, x = 0;
#pragma omp parallel for simd shared(x) num_threads(OMPVV_NUM_THREADS_HOST)
for (int i = 0; i < N; i++) {
#pragma omp atomic update
x += 1;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, x != N);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_parallel_for_simd_atomic());
OMPVV_REPORT_AND_RETURN(errors);
}
