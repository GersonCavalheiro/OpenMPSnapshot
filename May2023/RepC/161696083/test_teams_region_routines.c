#include <stdio.h>
#include <omp.h>
#include "ompvv.h"
int test_teams_region_routines()
{
int num_teams[OMPVV_NUM_TEAMS_DEVICE];
int num_threads[OMPVV_NUM_TEAMS_DEVICE];
int errors[2] = {0,0};
for (int x = 0; x < OMPVV_NUM_TEAMS_DEVICE; ++x) {
num_teams[x] = -99;
}
for (int x = 0; x < OMPVV_NUM_THREADS_DEVICE; ++x) {
num_threads[x] = -99;
}
omp_set_num_teams(OMPVV_NUM_TEAMS_DEVICE);
omp_set_teams_thread_limit(OMPVV_NUM_THREADS_DEVICE);
OMPVV_TEST_AND_SET_VERBOSE(errors[0], omp_get_max_teams() != OMPVV_NUM_TEAMS_DEVICE);
OMPVV_TEST_AND_SET_VERBOSE(errors[1], omp_get_teams_thread_limit() != OMPVV_NUM_THREADS_DEVICE);
#pragma omp teams 
{
num_teams[omp_get_team_num()] = omp_get_num_teams();
#pragma omp parallel masked
num_threads[omp_get_team_num()]= omp_get_num_threads();
}
OMPVV_ERROR_IF(num_teams[0] != OMPVV_NUM_TEAMS_DEVICE, "Test ran with invalid number of teams.");
OMPVV_ERROR_IF(num_threads[0] != OMPVV_NUM_THREADS_DEVICE, "Test returned an invalid number of threads.");
for (int x = 0; x < OMPVV_NUM_TEAMS_DEVICE; ++x) {
if (num_teams[x] != num_teams[0]) {
errors[0]++;
}
if (num_threads[x] != num_threads[0]) {
errors[1]++;
}
}
OMPVV_INFOMSG_IF(!(errors[0] + errors[1]), "Test passed with %d teams and %d threads per team.", num_teams[0], num_threads[0]);
return errors[0] + errors[1];
}
int main() {
int errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors,  test_teams_region_routines()!= 0);
OMPVV_REPORT_AND_RETURN(errors);
}
