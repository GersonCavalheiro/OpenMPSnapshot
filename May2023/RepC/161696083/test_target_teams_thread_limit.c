#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int errors[OMPVV_NUM_TEAMS_DEVICE];
int num_teams = 0;
int sum_errors = 0;
int i;
int testing_thread_limit = OMPVV_NUM_THREADS_DEVICE/OMPVV_NUM_TEAMS_DEVICE;
if (testing_thread_limit == 1)
testing_thread_limit = 2;
OMPVV_TEST_OFFLOADING;
for (i = 0; i<OMPVV_NUM_TEAMS_DEVICE; i++){
errors[i] = 0;
}
#pragma omp target map(tofrom:num_teams,errors) thread_limit(testing_thread_limit)
{
#pragma omp teams num_teams(OMPVV_NUM_TEAMS_DEVICE) 
{
#pragma omp parallel
{
if (omp_get_team_num() == 0 && omp_get_thread_num() == 0) {
num_teams = omp_get_num_teams();
}
if (omp_get_thread_num() == 0) {
if (omp_get_num_threads() > testing_thread_limit) {
errors[omp_get_team_num()] += 1;
}
}
}
}
}
for (i = 0; i<num_teams; i++){
sum_errors += errors[i];
}
OMPVV_WARNING_IF(num_teams != OMPVV_NUM_TEAMS_DEVICE, "The number of teams was unexpected, the test results are likely inconclusive")
OMPVV_WARNING_IF(testing_thread_limit == 1, "Only one thread was allocated to each team, the test results are likely inconclusive");
OMPVV_REPORT_AND_RETURN(sum_errors);
}
