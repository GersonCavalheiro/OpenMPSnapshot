#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int test_defaultmap_present_pointer() {
int errors = 0;
int i;
int *ptr;
int A[N];
for (i = 0; i < N; i++) {
A[i] = i;
}
#pragma omp target enter data map(to: ptr)
#pragma omp target map(tofrom: errors) defaultmap(present: pointer)
{  
ptr = &A[0];
for (i = 0; i < N; i++) {
if (ptr[i] != i) {
errors++;
}
ptr[i] = 2+i;
}
}
OMPVV_ERROR_IF(errors > 0, "Values were not mapped to the device properly");
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != 2+i);
}
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_present_pointer() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
