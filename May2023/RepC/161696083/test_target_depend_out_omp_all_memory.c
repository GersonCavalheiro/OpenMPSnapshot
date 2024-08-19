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
#pragma omp task shared(x, y) depend(out: omp_all_memory)
{
for(int i = 0; i < N; i++) {
x[i] += y;   
}
}
#pragma omp task shared(x, y, errors) depend(in: x, y)
{
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, x[i] != i + y);
}
}
#pragma omp taskwait
}
}
OMPVV_REPORT_AND_RETURN(errors);
}
