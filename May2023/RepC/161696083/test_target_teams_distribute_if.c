#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
int test_target_teams_distribute_if() {
OMPVV_INFOMSG("test_target_teams_distribute_if");
int isOffloading = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test if");
int a[N];
int errors = 0;
int attempt = 0;
int i;
for (int x = 0; x < N; ++x) {
a[x] = 1;
}
for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute if(attempt >= ATTEMPT_THRESHOLD) map(tofrom: a)
for (i = 0; i < N; ++i) {
if (attempt >= ATTEMPT_THRESHOLD) {
a[i] += (isOffloading && omp_is_initial_device() ? 10 : 0); 
} else {
a[i] += (omp_is_initial_device() ? 1 : 100);                
}
}
}
for (i = 0; i < N; ++i) {
OMPVV_TEST_AND_SET(errors, a[i] != (1 + ATTEMPT_THRESHOLD));
}
if (errors) {
int sum = 0;
for (i = 0; i < N; ++i) {
sum += a[i];
}
if (sum == N*(100*ATTEMPT_THRESHOLD + 1)) {
OMPVV_ERROR("Error in if. The execution was expected to occur on the host, but it occurred on the device.");
} else if (sum == N*(ATTEMPT_THRESHOLD + 10*(NUM_ATTEMPTS - ATTEMPT_THRESHOLD) + 1)) {
OMPVV_ERROR("Error in if. The execution was expected to occur on the device, but it occurred on the host.");
} else {
OMPVV_ERROR("Error in if. The execution occurred inconsistently on the host or on the device.");
}
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if());
OMPVV_REPORT_AND_RETURN(errors);
}
