#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int errors = 0;
int x[N];
int y = 5;
for (int i = 0; i < N; i++) {
x[i] = i;
}
OMPVV_TEST_OFFLOADING;
#pragma omp target map(tofrom: errors) map(to: x, y)
{
#pragma omp parallel
#pragma omp single
{
#pragma omp task depend(out: x)
{
for (int i = 0; i < N; i++) {
x[i] += 1;
}
}
#pragma omp task depend(out: y)
{   
y += 5;
}
#pragma omp task depend(inout: omp_all_memory)
{
for(int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, x[i] != i + 1);
}
OMPVV_TEST_AND_SET(errors, y != 10 );
}
#pragma omp task depend(out: x)
{
for (int i = 0; i < N; i++) {
x[i] += 1;
}
}
#pragma omp task depend(out: y)
{
y += 5;
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, x[i] != i + 2);
}
OMPVV_TEST_AND_SET(errors, y != 15);
}
OMPVV_REPORT_AND_RETURN(errors);
}
