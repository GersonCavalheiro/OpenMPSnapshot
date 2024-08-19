#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "ompvv.h"
#define N 1000
#define ARRAY_SIZE 5
int test_struct() {
OMPVV_INFOMSG("test_struct");
int errors = 0, isHost = -1;
int* pointers[ARRAY_SIZE + 1];
struct {
int a; 
int b[N]; 
int *p; 
} single, array[ARRAY_SIZE], singleCopy, arrayCopy[ARRAY_SIZE];
single.p = (int*) malloc(ARRAY_SIZE * sizeof(int));
pointers[0] = single.p;
single.a = 1;
for (int i = 0; i < N; ++i)
single.b[i] = i;
for (int i = 0; i < ARRAY_SIZE; ++i) {
array[i].p = (int*) malloc(ARRAY_SIZE * sizeof(int));
pointers[i + 1] = array[i].p;
array[i].a = 1;
for (int j = 0; j < N; ++j)
array[i].b[j] = j;
}
{
#pragma omp target enter data map(to: single) map(to: array[0:ARRAY_SIZE])
printf(""); 
}
#pragma omp target map(from: singleCopy) map(from: arrayCopy[0:ARRAY_SIZE]) map(tofrom: isHost)map(alloc: single, array[0:ARRAY_SIZE])
{
isHost = omp_is_initial_device();
singleCopy.a = single.a;
singleCopy.p = single.p;
for (int i = 0; i < N; ++i)
singleCopy.b[i] = single.b[i];
for (int i = 0; i < ARRAY_SIZE; ++i) {
arrayCopy[i].a = array[i].a;
arrayCopy[i].p = array[i].p;
for (int j = 0; j < N; ++j)
arrayCopy[i].b[j] = array[i].b[j];
}
}
OMPVV_TEST_AND_SET(errors, (singleCopy.a != single.a)); 
for (int i = 0; i < N; ++i)
OMPVV_TEST_AND_SET(errors, (singleCopy.b[i] != single.b[i]));
OMPVV_TEST_AND_SET(errors, (pointers[0] != singleCopy.p));
for (int i = 0; i < ARRAY_SIZE; ++i) {
OMPVV_TEST_AND_SET(errors, (arrayCopy[i].a != array[i].a)); 
for (int j = 0; j < N; ++j)
OMPVV_TEST_AND_SET(errors, (arrayCopy[i].b[j] != array[i].b[j]));
OMPVV_TEST_AND_SET(errors, (pointers[i + 1] != arrayCopy[i].p));
}
#pragma omp target exit data map(delete: single, array[0:ARRAY_SIZE])
free(single.p);
for (int i = 0; i < ARRAY_SIZE; ++i) {
free(array[i].p);
}
return errors;
}
int test_typedef() {
OMPVV_INFOMSG("test_typedef");
int errors = 0, isHost = -1;
int* pointers[ARRAY_SIZE + 1];
typedef struct {
int a;
int b[N];
int *p;
} test_struct;
test_struct single, array[ARRAY_SIZE], singleCopy, arrayCopy[ARRAY_SIZE];
single.p = (int*) malloc(ARRAY_SIZE * sizeof(int));
pointers[0] = single.p;
single.a = 1;
for (int i = 0; i < N; ++i)
single.b[i] = i;
for (int i = 0; i < ARRAY_SIZE; ++i) {
array[i].p = (int*) malloc(ARRAY_SIZE * sizeof(int));
pointers[i + 1] = array[i].p;
array[i].a = 1;
for (int j = 0; j < N; ++j)
array[i].b[j] = j;
}
{
#pragma omp target enter data map(to: single) map(to: array[0:ARRAY_SIZE])
printf(""); 
}
#pragma omp target map(from: singleCopy) map(from: arrayCopy[0:ARRAY_SIZE]) map(tofrom: isHost) map(alloc: single, array[0:ARRAY_SIZE])
{
isHost = omp_is_initial_device();
singleCopy.a = single.a;
singleCopy.p = single.p;
for (int i = 0; i < N; ++i)
singleCopy.b[i] = single.b[i];
for (int i = 0; i < ARRAY_SIZE; ++i) {
arrayCopy[i].a = array[i].a;
arrayCopy[i].p = array[i].p;
for (int j = 0; j < N; ++j)
arrayCopy[i].b[j] = array[i].b[j]; 
}
}
OMPVV_TEST_AND_SET(errors, (singleCopy.a != single.a));
for (int i = 0; i < N; ++i)
OMPVV_TEST_AND_SET(errors, (singleCopy.b[i] != single.b[i]));
OMPVV_TEST_AND_SET(errors, (pointers[0] != singleCopy.p));
for (int i = 0; i < ARRAY_SIZE; ++i) {
OMPVV_TEST_AND_SET(errors, (arrayCopy[i].a != array[i].a)); 
for (int j = 0; j < N; ++j)
OMPVV_TEST_AND_SET(errors, (arrayCopy[i].b[j] != array[i].b[j]));
OMPVV_TEST_AND_SET(errors, (pointers[i + 1] != arrayCopy[i].p));
}
#pragma omp target exit data map(delete: single, array[0:ARRAY_SIZE])
free(single.p);
for (int i = 0; i < ARRAY_SIZE; ++i) {
free(array[i].p);
}
return errors;
}
int main () {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET(errors, test_struct());
OMPVV_TEST_AND_SET(errors, test_typedef());
OMPVV_REPORT_AND_RETURN(errors);
}
