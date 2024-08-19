#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1028
#define STRIDE_LEN 100
int test_simd_nontemporal() {
int errors = 0;
int i;
int a[N], b[N], c[N];
for (i = 0; i < N; i++) {
a[i] = 10; 
b[i] = i;
c[i] = 2 * i;
}   
#pragma omp simd nontemporal (a, b, c)
for (i = 0; i < N; i += STRIDE_LEN) {
a[i] = b[i] * c[i];
}   
for (i = 0; i < N; i++) {
if (i % STRIDE_LEN == 0) { 
OMPVV_TEST_AND_SET(errors, a[i] != (b[i] * c[i]));
} else { 
OMPVV_TEST_AND_SET(errors, a[i] != 10);
}
}   
return errors;
}
int main () {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_simd_nontemporal())
OMPVV_REPORT_AND_RETURN(errors);
}
