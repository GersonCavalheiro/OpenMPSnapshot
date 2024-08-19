#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int errors = 0;
int x[N];
int y[N];
for (int i = 0; i < N; i++) {
x[i] = i;
y[i] = i;
}
OMPVV_TEST_OFFLOADING;
#pragma omp target map(tofrom: x,y)
{
#pragma omp parallel 
{
#pragma omp loop order(reproducible:concurrent) 
for (int i = 0; i < N; i++) {
x[i] = x[i] + 2;	
}
#pragma omp loop order(reproducible:concurrent)
for (int i = 0; i < N; i++) {
y[i] = x[i] + 2;
}
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, y[i] != i + 4);
}
OMPVV_REPORT_AND_RETURN(errors);
}
