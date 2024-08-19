#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int test_map_struct() {
OMPVV_INFOMSG("Running test_map_struct()");
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
{
#pragma omp target
if (!omp_is_initial_device()) {
single.a = 1;
for (int i = 0; i < N; ++i)
single.b[i] = 1;
for (int i = 0; i < 5; ++i) {
array[i].a = 1;
for (int j = 0; j < N; ++j)
array[i].b[j] = 1;
}
}
} 
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
return errors;
}
int test_map_typedef() {
OMPVV_INFOMSG("Running test_map_typedef()");
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
{
#pragma omp target
if (!omp_is_initial_device()) {  
single.a = 1;
for (int i = 0; i < N; ++i)
single.b[i] = 1;
for (int i = 0; i < 5; ++i) {
array[i].a = 1;
for (int j = 0; j < N; ++j)
array[i].b[j] = 1;
}
}
} 
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
return errors;
}
int main () {
int errors = 0;
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
if (!is_offloading) {
OMPVV_ERROR("Mapping to device cannot be properly tested as offloading is not enabled");
errors ++;
OMPVV_REPORT_AND_RETURN(errors);
}
errors += test_map_struct();
errors += test_map_typedef();
OMPVV_REPORT_AND_RETURN(errors);
}
