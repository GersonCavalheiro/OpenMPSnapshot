#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <omp.h>
#define N 10
int A[N] = {0};
int B[N] = {0};
int test_tofrom() {
int errors = 0;
for (int i = 0; i < N; ++i) {
A[i] = 0;
}
#pragma omp target enter data map(to: A)
#pragma omp target
{
for (int i = 0; i < N; i++) {
A[i] = N;
}
}
#pragma omp target exit data map(from: A)
for (int i = 0; i < N; ++i) {
OMPVV_TEST_AND_SET_VERBOSE(errors, A[i] != N);
}
return errors;
}
int test_delete() {
int errors = 0;
for (int i = 0; i < N; ++i) {
A[i] = N;
}
#pragma omp target data map(tofrom: A) map(from: B)
{
#pragma omp target exit data map(delete: A)
for (int i = 0; i < N; ++i) {
A[i] = 0;
}
#pragma omp target map(to: A)   
{
for (int i = 0; i < N; ++i) {
B[i] = A[i];
}
}
}
for (int i = 0; i < N; ++i) {
OMPVV_TEST_AND_SET_VERBOSE(errors, B[i] != 0);
}
return errors;
}
int main () {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_SHARED_ENVIRONMENT;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_tofrom() != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_delete() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
