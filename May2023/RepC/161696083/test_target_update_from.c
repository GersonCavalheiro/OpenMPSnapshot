#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 100
int a[N];
int b[N];
int c[N];
int main() {
int errors = 0, i = 0, change_flag = 0;
for (i = 0; i < N; i++) {
a[i] = 10;
b[i] = 2; 
}
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
#pragma omp target data map(to: a[:N], b[:N]) 
{
#pragma omp target
{
int j = 0;
for (j = 0; j < N; j++) {
b[j] = (a[j] + b[j]);
}
} 
#pragma omp target update from(b[:N]) 
} 
for (i = 0; i < N; i++) 
OMPVV_TEST_AND_SET_VERBOSE(errors, (b[i] != 12)); 
#pragma omp target 
{
int j = 0;
for (j = 0; j < N; j++) {
c[j] = (2* b[j]);
}
} 
for (i = 0; i < N; i++) 
OMPVV_TEST_AND_SET_VERBOSE(errors, (c[i] != 24)); 
OMPVV_REPORT_AND_RETURN(errors);
}
