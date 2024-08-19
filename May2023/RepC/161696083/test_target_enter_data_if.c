#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define SIZE_THRESHOLD 512
#define ARRAY_SIZE 1024
int main() {
int a[ARRAY_SIZE];
int b[ARRAY_SIZE];
int c[ARRAY_SIZE];
int size, i = 0, errors = 0, isOffloading = 0, isSharedMemory = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedMemory)
if (!isOffloading || isSharedMemory) {
OMPVV_WARNING("It is not possible to test conditional data transfers "
"if the environment is shared or offloading is off. Not testing "
"anything")
OMPVV_REPORT_AND_RETURN(0);
}
for (size = 256; size <= ARRAY_SIZE; size += 256) {
for (i = 0; i < size; i++) {
a[i] = 1;
b[i] = i;
c[i] = -1;
}
#pragma omp target enter data if(size > SIZE_THRESHOLD) map(to: size) map (to: a[0:size], b[0:size])
for (i = 0; i < size; i++) {
a[i] = 0;
b[i] = 0;
}
#pragma omp target map(tofrom: a[0:size], b[0:size], c[0:size])
{
int j = 0;
for (j = 0; j < size; j++) {
c[j] = a[j] + b[j];
}
} 
for (i = 0; i < size; i++) {
if (size > SIZE_THRESHOLD) {
OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != i + 1)
} else {
OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != 0)
} 
}
#pragma omp target exit data if(size > SIZE_THRESHOLD) map(delete: a[0:size], b[0:size])
} 
OMPVV_REPORT_AND_RETURN(errors)
}
