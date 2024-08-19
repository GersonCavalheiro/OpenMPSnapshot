#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024
int checkPreconditions() {
int isOffloading = 0;
int i;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test if");
return isOffloading;
}
int test_target_teams_distribute_if_target_modifier() {
OMPVV_INFOMSG("test_target_teams_distribute_if_target_modifier");
int a[N], warning[N];
int attempt = 0;
int errors = 0;
int i;
int isOffloading;
isOffloading = checkPreconditions();
for (i = 0; i < N; i++) {
a[i] = 1;
warning[i] = 0;
}
for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute parallel for if(target: attempt >= ATTEMPT_THRESHOLD)map(tofrom: a) num_threads(OMPVV_NUM_THREADS_DEVICE)
for (i = 0; i < N; i++) {
warning[i] += (omp_get_num_threads() == 1) ? 1 : 0; 
if (attempt >= ATTEMPT_THRESHOLD) {
a[i] += (isOffloading && omp_is_initial_device() ? 10 : 0); 
}
else {
a[i] += (omp_is_initial_device() ? 1 : 100);
} 
}
}
int raiseWarning = 0;
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != 1 + (ATTEMPT_THRESHOLD));
if (warning[i] != 0) {
raiseWarning = 1;
}
}
OMPVV_WARNING_IF(raiseWarning != 0, "The number of threads was 1 even though we expected it to be more than 1. Not a compliance error in the specs");
OMPVV_ERROR_IF(errors, "error in if(target: modifier). The execution was expected to occur in the device, but it happened in the host when if(false), or the other way around");
return errors;
}
int main() {
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if_target_modifier());
OMPVV_REPORT_AND_RETURN(errors);
}
