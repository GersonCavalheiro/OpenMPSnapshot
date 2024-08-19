#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 256
int test_uses_allocators_pteam() {
int errors = 0;
int pteam_result[N] = {0};
int device_result[N] = {0};
int result[N] = {0};
for (int i = 0; i < N; i++) {
result[i] = 2 * i ;
}
#pragma omp target parallel for map(from: device_result) uses_allocators(omp_pteam_mem_alloc) allocate(omp_pteam_mem_alloc: pteam_result) private(pteam_result)
for (int i = 0; i < N; i++) {
pteam_result[i] = 2 * i ;
device_result[i] = pteam_result[i];
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET_VERBOSE(errors, result[i] != device_result[i]);
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_uses_allocators_pteam() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
