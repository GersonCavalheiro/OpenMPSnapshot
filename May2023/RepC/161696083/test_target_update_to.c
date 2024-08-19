#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 1024
int a[N];
int b[N];
int c[N];
void update_b() {
int i;
for (i = 0; i < N; i++) {
b[i] = b[i] * 2;
}
}
int main() {
int errors= 0;
int i = 0;
OMPVV_TEST_OFFLOADING;
for (i = 0; i < N; i++) {
a[i] = 10;
b[i] = 2;
c[i] = 0;
}
#pragma omp target data map(to: a[:N], b[:N]) map(from: c)
{
#pragma omp target
{
int j = 0;
for (j = 0; j < N; j++) {
c[j] = (a[j] + b[j]);        
}
}
update_b();
#pragma omp target update to(b[:N])  
#pragma omp target
{
int j = 0;
for (j = 0; j < N; j++) {
c[j] = (c[j] + b[j]);        
}
}
}
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != 16);
}
OMPVV_REPORT_AND_RETURN(errors);
}
