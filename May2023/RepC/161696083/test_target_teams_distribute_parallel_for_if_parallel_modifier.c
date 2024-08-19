#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024
void checkPreconditions() {
int isOffloading = 0;
int i;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "With offloading off, it is not possible to test that if(parallel:) is not affecting the target offloading");
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
OMPVV_WARNING_IF(raiseWarningHost == 0, "Initial number of threads in host was 1. It is not possible to test the if for parallel");
}
int test_target_teams_distribute_if_parallel_modifier() {
OMPVV_INFOMSG("test_target_teams_distribute_if_parallel_modifier");
int a[N];
int warning[N] ; 
int attempt = 0;
int errors = 0;
int i;
checkPreconditions();
for (i = 0; i < N; i++) {
a[i] = 0;
warning[i] = 0;
}
for (attempt = 0; attempt < NUM_ATTEMPTS; ++attempt) {
#pragma omp target teams distribute parallel for if(parallel: attempt >= ATTEMPT_THRESHOLD)map(tofrom: a, warning) num_threads(OMPVV_NUM_THREADS_DEVICE)
for (i = 0; i < N; i++) {
if (omp_is_initial_device())
a[i] += 10; 
if (attempt >= ATTEMPT_THRESHOLD) {
if (omp_get_num_threads() == 1) {
warning[i] += 1;
}
} else {
a[i] += (omp_get_num_threads() != 1) ? 10 : 1;  
}
}
}
int raiseWarning = 0;
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != ATTEMPT_THRESHOLD);
if (warning[i] != 0) {
raiseWarning++;
}
}
OMPVV_WARNING_IF(raiseWarning == N * (NUM_ATTEMPTS - ATTEMPT_THRESHOLD), "The number of threads was 1 when a number > 1 was expected. if(parallel:true). Not a compliance error in the specs");
OMPVV_ERROR_IF(errors, "error in if(parallel: modifier). Possible causes are: the execution occurred in the host even though it should not affect the target region. The number of threads was > 1 when if(false).");
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_if_parallel_modifier());
OMPVV_REPORT_AND_RETURN(errors);
}
