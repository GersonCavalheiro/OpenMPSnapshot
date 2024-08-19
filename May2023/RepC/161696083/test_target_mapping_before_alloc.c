#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int to_before_alloc() {
int i;
int errors = 0;
int scalar = 80;
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
#pragma omp target map(alloc: scalar, a, member) map(to: scalar, a, member) map(tofrom: errors) 
{
if (scalar != 80 || a[2] != 2 || member.var != 1 || member.b[2] != 2) {
errors++;
}	
}
return errors;
}
int main () {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, to_before_alloc());
OMPVV_REPORT_AND_RETURN(errors);
}
