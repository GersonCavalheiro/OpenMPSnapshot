#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ompvv.h"
#define N 1000
int test_math_lib_inside_target() {
OMPVV_INFOMSG("test_math_lib_inside_target");
double array[N];
int errors = 0;
for (int i = 0; i < N; ++i) {
array[i] = 0.99;
}
int c99_zero = FP_ZERO;
#pragma omp target map(tofrom: array[0:N]) 
for (int i = 0; i < N; ++i) {
array[i] = pow((double)i,2.0);
}
for (int i = 0; i < N; ++i) {
OMPVV_TEST_AND_SET(errors, (array[i] - pow((double)i,2)) > 0.000009);
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_math_lib_inside_target());
OMPVV_REPORT_AND_RETURN(errors);
}
