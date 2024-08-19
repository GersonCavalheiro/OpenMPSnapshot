#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define ARRAY_SIZE 1024
int test_target_simd_simdlen() {
OMPVV_INFOMSG("test_target_simd_simdlen");
OMPVV_WARNING("This test cannot check if actual SIMD extensions at the hardware level" \
" were used, or of the generated code is different in any way");
int errors = 0;
int A[ARRAY_SIZE];
int i, len;
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] = 1;
}
#pragma omp target simd simdlen(1) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
#pragma omp target simd simdlen(5) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
#pragma omp target simd simdlen(8) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
#pragma omp target simd simdlen(13) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
#pragma omp target simd simdlen(16) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
#pragma omp target simd simdlen(100) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
#pragma omp target simd simdlen(128) map(tofrom: A[0:ARRAY_SIZE])
for (i = 0; i < ARRAY_SIZE; ++i) {
A[i] += A[i]; 
}
for (i = 0; i < ARRAY_SIZE; ++i) {
OMPVV_TEST_AND_SET(errors, A[i] != 1<<7);
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_simd_simdlen());
OMPVV_REPORT_AND_RETURN(errors);
}
