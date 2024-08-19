#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_allocate() {
int errors = 0;
int x[N];
#pragma omp allocate(x) 
#pragma omp parallel for 
for (int i = 0; i < N; i++) {
x[i] = i;
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, x[i] != i);
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_allocate() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
