#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
#pragma omp requires unified_shared_memory
int unified_shared_memory_target_alloc() {
OMPVV_INFOMSG("Unified shared memory testing - Memory allocated in the device");
int errors = 0;
int *anArray = (int *) omp_target_alloc(sizeof(int)*N, omp_get_default_device());
int anArrayCopy[N];
#pragma omp target 
{
for (int i = 0; i < N; i++) {
anArray[i] = i;
}
}
for (int i = 0; i < N; i++) {
anArrayCopy[i] = 0;
}
#pragma omp target
{
for (int i = 0; i < N; i++) {
anArray[i] += 10;
}
}
for (int i = 0; i < N; i++) {
anArray[i] += 10;
}
#pragma omp target
{
for (int i = 0; i < N; i++) {
anArrayCopy[i] = anArray[i];
}
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, anArray[i] != i + 20);
OMPVV_TEST_AND_SET_VERBOSE(errors, anArrayCopy[i] != i + 20);
if (errors) break;
}
omp_target_free(anArray,omp_get_default_device());
return errors;
}
int main() {
int isOffloading;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "With no offloading, unified shared memory is guaranteed due to host execution");
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, unified_shared_memory_target_alloc());
OMPVV_REPORT_AND_RETURN(errors);
}
