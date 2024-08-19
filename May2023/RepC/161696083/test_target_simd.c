#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define ARRAY_SIZE 1024
int test_target_simd() {
OMPVV_INFOMSG("test_target_simd");
OMPVV_WARNING("This test cannot check if actual SIMD extensions at the hardware level"
" were used, or of the generated code is different in any way");
int errors = 0;
int a[ARRAY_SIZE];
int b[ARRAY_SIZE];
int c[ARRAY_SIZE];
int i;
for (i = 0; i < ARRAY_SIZE; ++i) {
a[i] = 1;
b[i] = i;
c[i] = 2 * i;
}
#pragma omp target simd map(to: b[0:ARRAY_SIZE], c[0:ARRAY_SIZE]) map(tofrom: a[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
a[i] += b[i] * c[i];
}
for (i = 0; i < ARRAY_SIZE; ++i) {
OMPVV_TEST_AND_SET(errors, (a[i] != 1 + (b[i] * c[i])));
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_simd());
OMPVV_REPORT_AND_RETURN(errors);
}
