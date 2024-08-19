#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int errors; 
int test_present_map_reordering() {
int x[N];
for (int i = 0; i < N; i++) { 
x[i] = i;
}
#pragma omp target data map(tofrom: x) 
{
#pragma omp target map(present, to: x) map(from: x)
{
for (int i = 0; i < N; i++) {
x[i] += i;
}
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, x[i] != i*2);
}
return errors;  	 
}
int main () {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_present_map_reordering());
OMPVV_REPORT_AND_RETURN(errors);
}
