#include <stdlib.h>
#include <stdio.h>
#include <omp.h> 
#include "ompvv.h" 
#define N 1024
int test_simd_if () {
int errors = 0;
int i;
int b[N], c[N];
struct new_struct {
int a[N];
};
struct new_struct struct_t;
for (i = 0; i < N; i++ ) {
struct_t.a[i] = i;
b[i] = i + 5;
c[i] = 0;
}
int k = N;
#pragma omp target simd simdlen(64) if(k == N) 
for (i = 0; i < N; i++) {
c[i] = struct_t.a[i] * b[i];
}
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != (struct_t.a[i] * b[i]));
}
#pragma omp target simd simdlen(64) if(k != N) 
for (i = 0; i < N; i++) {
c[i] = struct_t.a[i] * 2 * b[i];
}
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != (struct_t.a[i] * 2 * b[i]));
}
return errors;
}
int main () { 
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_WARNING("Vectorization resulting from the construct cannot be guaranteed"); 
OMPVV_TEST_AND_SET_VERBOSE(errors, test_simd_if());
OMPVV_REPORT_AND_RETURN(errors);
}
