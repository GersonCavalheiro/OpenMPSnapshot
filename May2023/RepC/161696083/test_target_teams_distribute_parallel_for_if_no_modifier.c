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
OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test if on the parallel and not the target");
int init_num_threads_dev[N], init_num_threads_host[N];
for (i = 0; i < N; i++) {
init_num_threads_dev[i] = 0;
init_num_threads_host[i] = 0;
}
#pragma omp target teams distribute parallel for num_threads(OMPVV_NUM_THREADS_DEVICE)
for (i = 0; i < N; i++) {
init_num_threads_dev[i] = omp_get_num_threads();
}
#pragma omp parallel for num_threads(OMPVV_NUM_THREADS_DEVICE)
for (i = 0; i < N; i++) {
init_num_threads_host[i] = omp_get_num_threads();
}
int raiseWarningDevice = 0, raiseWarningHost = 0;
for (i = 0; i < N; i++) {
if (init_num_threads_dev[i] > 1 ) {
raiseWarningDevice +=  1;
}
if ( init_num_threads_host[i] > 1) {
raiseWarningHost += 1;
}
}
OMPVV_WARNING_IF(raiseWarningDevice == 0, "Initial number of threads in device was 1. It is not possible to test the if for the parallel directive");
OMPVV_WARNING_IF(raiseWarningHost == 0, "Initial number of threads in host was 1. It is not possible to test the if for the parallel directive");
return isOffloading;
}
int test_target_teams_distribute_if_no_modifier() {
OMPVV_INFOMSG("test_target_teams_distribute_if_no_modifier");
int isOffloading = checkPreconditions();
int a[N];
int warning[N] ; 
int attempt = 0;
int errors = 0;
int i;
for (i = 0; i < N; i++) {
a[i] = 1;
warning[i] = 0;
}
for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute parallel for if(attempt >= ATTEMPT_THRESHOLD)map(tofrom: a, warning) num_threads(OMPVV_NUM_THREADS_DEVICE)
for (i = 0; i < N; i++) {
if (omp_is_initial_device()) {
a[i] += (omp_get_num_threads() > 1) ? 10 : 0; 
a[i] += (attempt >= ATTEMPT_THRESHOLD) ? 10 : 0; 
} else {
a[i] += 1;
warning[i] += (omp_get_num_threads() == 1 ? 1 : 0); 
}
}
}
int raiseWarning = 0;
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != 1 + (NUM_ATTEMPTS - ATTEMPT_THRESHOLD));
if (warning[i] != 0) {
raiseWarning = 1;
}
}
OMPVV_WARNING_IF(raiseWarning != 0, "The number of threads was 1 even though we expected it to be more than 1. Not a compliance error in the specs");
OMPVV_ERROR_IF(errors, "error in if(no-modifier). Possible causes are: the number of threads was greater than 1 for if(false), the test executed in the host for if(true), or the test executed in the device for if(false)");
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if_no_modifier());
OMPVV_REPORT_AND_RETURN(errors);
}
