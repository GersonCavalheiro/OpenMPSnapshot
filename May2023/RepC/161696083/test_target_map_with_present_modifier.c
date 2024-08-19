#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int errors; 
int test_present_modifier() {
int i;
int scalar = 1;
int a[N];
struct {
int var;
int b[N];
} member; 
member.var = 1;
for (i = 0; i < N; i++) { 
a[i] = i;
member.b[i] = i;
}
#pragma omp target data map(tofrom:scalar, a, member) 
{
#pragma omp target map(present, tofrom: scalar, a, member) 
{
scalar += 1;
member.var += 2;
for (i = 0; i < N; i++) {
a[i] += i;
member.b[i] += i;
}
}
}
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != i*2);
OMPVV_TEST_AND_SET(errors, member.b[i] != i*2);
}
OMPVV_TEST_AND_SET(errors, scalar != 2);
OMPVV_TEST_AND_SET(errors, member.var != 3);
return errors;  	 
}
int main () {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_present_modifier());
OMPVV_REPORT_AND_RETURN(errors);
}
