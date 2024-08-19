#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ompvv.h"
#define N 1000
int test_async_between_task_target() {
OMPVV_INFOMSG("test_async_between_task_target");
int errors = 0;
double sum = 0.0;
double* h_array = (double *) malloc(N * sizeof(double));
double* in_1 = (double *) malloc(N * sizeof(double));
double* in_2 = (double *) malloc(N * sizeof(double));
#pragma omp task depend(out: in_1) shared(in_1)
{
for (int i = 0; i < N; ++i) {
in_1[i] = 1;
}
}
#pragma omp task depend(out: in_2) shared(in_2)
{
for (int i = 0; i < N; ++i) {
in_2[i] = 2;
}
}
#pragma omp target enter data map(alloc: h_array[0:N]) map(to: in_1[0:N]) map(to: in_2[0:N]) depend(out: h_array) depend(in: in_1) depend(in: in_2) 
#pragma omp task shared (h_array, in_1, in_2) depend(inout: h_array) depend(in: in_1) depend(in: in_2)
{
#pragma omp target  
{
for (int i = 0; i < N; ++i) {
h_array[i] = in_1[i]*in_2[i];
}
}
}
#pragma omp target exit data map(from: h_array[0:N]) depend(inout: h_array) 
#pragma omp task depend(in: h_array) shared(sum, h_array)
{
for (int i = 0; i < N; ++i) {
sum += h_array[i];
}
}
#pragma omp taskwait
errors = 2.0*N != sum;
#pragma omp target exit data map(release: h_array[0:N], in_1[0:N], in_2[0:N])
free(h_array);
free(in_1);
free(in_2);
return errors;
}
int test_async_between_target() {
OMPVV_INFOMSG("test_async_between_target");
int errors = 0;
int sum = 0;
int* h_array = (int *) malloc(N * sizeof(int));
int val = 2;
#pragma omp target enter data map(alloc: h_array[0:N]) depend(out: h_array) 
#pragma omp target enter data map(to: val) depend(out: val) 
#pragma omp target depend(inout: h_array) depend(in: val) 
{
for (int i = 0; i < N; ++i) {
h_array[i] = val;
}
}
#pragma omp target exit data map(from: h_array[0:N]) depend(in: h_array) 
#pragma omp taskwait
for (int i = 0; i < N; ++i) {
sum += h_array[i];
}
OMPVV_TEST_AND_SET(errors, 2*N != sum);
#pragma omp target exit data map(release: h_array[0:N], val)
free(h_array);
return errors;
}
int main(){
int errors = 0;
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_async_between_target());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_async_between_task_target());
OMPVV_REPORT_AND_RETURN(errors);
}
