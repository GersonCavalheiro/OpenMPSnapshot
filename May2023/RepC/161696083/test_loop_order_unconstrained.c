#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int errors = 0;
int x[N];
for (int i = 0; i < N; i++) {
x[i] = i;
}
OMPVV_TEST_OFFLOADING;
#pragma omp target map(tofrom: x)
{
#pragma omp parallel loop order(unconstrained:concurrent) 
for (int i = 0; i < N; i++) {
x[i] = x[i] + 2;	
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, x[i] != i + 2);
}
OMPVV_REPORT_AND_RETURN(errors);
}
