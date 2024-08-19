#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 1024
int test_defaultmap_present_aggregate() {
int errors = 0;
int i;
int a[N];
struct test_struct {
int s; 
int S[2];
};
for (i = 0; i < N; i++) {
a[i] = i;
}
struct test_struct new_struct;
new_struct.s = 10; new_struct.S[0] = 100; new_struct.S[1] = 1000;
#pragma omp target data map(tofrom: a, new_struct)
{
#pragma omp target map(tofrom: errors) defaultmap(present: aggregate)
{  
for (i = 0; i < N; i++) {
if(a[i] != i){errors++;}
a[i] += 2;
}
if(new_struct.s != 10 || new_struct.S[0] != 100 || new_struct.S[1] != 1000) {errors++;}
new_struct.s = 7; new_struct.S[0] = 70; new_struct.S[1] = 700;
}
}
OMPVV_ERROR_IF(errors > 0, "Values were not mapped to the device properly");
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != 2+i);
}
OMPVV_TEST_AND_SET(errors, new_struct.s != 7 ||  new_struct.S[0] != 70 || new_struct.S[1] != 700);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_present_aggregate() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
