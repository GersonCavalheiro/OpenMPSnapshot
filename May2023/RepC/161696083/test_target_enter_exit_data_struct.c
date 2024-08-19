#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "ompvv.h"
#define N 1000
int test_struct() {
OMPVV_INFOMSG("Running test_struct()");
int errors = 0;
int* pointers[6];
struct {
int a; 
int b[N]; 
int *p; 
} single, array[5];
single.p = (int*) malloc(5 * sizeof(int));
pointers[0] = single.p;
for (int i = 0; i < 5; ++i) {
array[i].p = (int*) malloc(5 * sizeof(int));
pointers[i + 1] = array[i].p;
}
#pragma omp target enter data map(to: single) map(to: array[0:5])
#pragma omp target map(alloc: single) map(alloc: array[0:5])
{
single.a = 1;
for (int i = 0; i < N; ++i)
single.b[i] = 1;
for (int i = 0; i < 5; ++i) {
array[i].a = 1;
for (int j = 0; j < N; ++j)
array[i].b[j] = 1;
}
} 
#pragma omp target exit data map(from: single) map(from: array[0:5])
OMPVV_TEST_AND_SET_VERBOSE(errors, (single.a != 1)); 
for (int i = 0; i < N; ++i)
OMPVV_TEST_AND_SET_VERBOSE(errors, (single.b[i] != 1));
OMPVV_TEST_AND_SET_VERBOSE(errors, (pointers[0] != single.p));
for (int i = 0; i < 5; ++i) {
OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].a != 1)); 
for (int j = 0; j < N; ++j)
OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].b[j] != 1));
OMPVV_TEST_AND_SET_VERBOSE(errors, (pointers[i + 1] != array[i].p));
}
free(single.p);
for (int i = 0; i < 5; ++i) {
free(array[i].p);
}
return errors;
}
int test_typedef() {
OMPVV_INFOMSG("Running test_typedef()");
int errors = 0;
int* pointers[6];
typedef struct {
int a;
int b[N];
int *p;
} test_struct;
test_struct single, array[5];
single.p = (int*) malloc(5 * sizeof(int));
pointers[0] = single.p;
for (int i = 0; i < 5; ++i) {
array[i].p = (int*) malloc(5 * sizeof(int));
pointers[i + 1] = array[i].p;
}
#pragma omp target enter data map(to: single) map(to: array[0:5])
#pragma omp target map(alloc: single) map(alloc: array[0:5])
{
single.a = 1;
for (int i = 0; i < N; ++i)
single.b[i] = 1;
for (int i = 0; i < 5; ++i) {
array[i].a = 1;
for (int j = 0; j < N; ++j)
array[i].b[j] = 1;
}
} 
#pragma omp target exit data map(from: single) map(from: array[0:5])
OMPVV_TEST_AND_SET_VERBOSE(errors, (single.a != 1)); 
for (int i = 0; i < N; ++i)
OMPVV_TEST_AND_SET_VERBOSE(errors, (single.b[i] != 1));
errors |= (pointers[0] != single.p);
for (int i = 0; i < 5; ++i) {
OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].a != 1)); 
for (int j = 0; j < N; ++j)
OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].b[j] != 1));
OMPVV_TEST_AND_SET_VERBOSE(errors, (pointers[i + 1] != array[i].p));
}
free(single.p);
for (int i = 0; i < 5; ++i) {
free(array[i].p);
}
return errors;
}
int main () {
int isOffloading;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
int errors = 0;
errors += test_struct();
errors += test_typedef();
OMPVV_REPORT_AND_RETURN(errors);
}
