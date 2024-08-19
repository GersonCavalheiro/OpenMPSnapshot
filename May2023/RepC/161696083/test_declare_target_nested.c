#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
#pragma omp declare target
int a[N], b[N], c[N]; 
int errors = 0;
int i = 0;
#pragma omp declare target
int test_target();
#pragma omp end declare target
#pragma omp end declare target
int test_target() { 
#pragma omp parallel for 
for (i = 0; i < N; i++) {
a[i] = 5;
b[i] = 10;
c[i] = 15;
}
for (i = 0; i < N; i++) {
if ( a[i] != 5 || b[i] != 10 || c[i] != 15) {
errors++;  
} 
}
return errors; 
} 
int test_wrapper() { 
#pragma omp target 
{
test_target();
}
#pragma omp target update from(errors, a, b, c)
for (i = 0; i < N; i++) { 
if ( a[i] != 5 || b[i] != 10 || c[i] != 15) {
errors++;  
} 
}
return errors;
}
int main () {
for (i = 0; i < N; i++) {
a[i] = i;
b[i] = 2*i;
c[i] = 3*i;
}
#pragma omp target update to(a,b,c) 
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_wrapper());
OMPVV_REPORT_AND_RETURN(errors);
}  
