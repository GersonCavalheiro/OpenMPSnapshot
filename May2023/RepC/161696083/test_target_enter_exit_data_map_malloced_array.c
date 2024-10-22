#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "ompvv.h"
#define N 10
int *x;
int *y;
int test_tofrom() {
int i, errors = 0;
int *A;
A = (int *) malloc(N*sizeof(int));
if (NULL == A) {
OMPVV_ERROR("Malloc returned NULL.");
exit(-1);
}
for (i = 0; i < N; ++i) {
A[i] = 0;
}
x = A;
#pragma omp target enter data map(to: x[:N])
#pragma omp target
{
for (i = 0; i < N; i++) {
x[i] = N;
}
}
#pragma omp target exit data map(from: x[:N])
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != N);
}
free(A);
return errors;
}
int test_delete() {
int i, errors = 0;
int *A, *B;
A = (int *) malloc(N*sizeof(int));
B = (int *) malloc(N*sizeof(int));
if (NULL == A || NULL == B) {
OMPVV_ERROR("Malloc returned NULL.");
exit(-1);
}
for (i = 0; i < N; ++i) {
A[i] = N;
}
x = A;
y = B;
#pragma omp target data map(tofrom: x[:N]) map(from: y[:N])
{
#pragma omp target exit data map(delete: x[:N])
for (i = 0; i < N; ++i) {
x[i] = 0;
}
#pragma omp target map(to: x[:N])
{
for (i = 0; i < N; i++) {
y[i] = x[i];
}
}
}
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, B[i] != 0);
}
free(A);
free(B);
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_tofrom());
OMPVV_TEST_AND_SET_VERBOSE(errors, test_delete());
OMPVV_REPORT_AND_RETURN(errors);
}
