#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define SIZE 1024
int main() {
int is_offloading = 0;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
int a[SIZE];
int share = 0;
int errors = 0;
int num_teams;
for (int x = 0; x < SIZE; ++x) {
a[x] = x;
}
#pragma omp target teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) shared(share, num_teams) map(to: a[0:SIZE]) defaultmap(tofrom:scalar)
for (int x = 0; x < SIZE; ++x) {
#pragma omp atomic write
num_teams = omp_get_num_teams();
#pragma omp atomic
share = share + a[x];
}
for (int x = 0; x < SIZE; ++x) {
share = share - x;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (share != 0));
OMPVV_ERROR_IF(errors != 0, "The value of share is = %d but expected 0.", share);
share = 5;
#pragma omp target data map(tofrom: a[0:SIZE]) map(tofrom: share)
{
#pragma omp target teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) shared(share)
for (int x = 0; x < SIZE; ++x) {
a[x] = a[x] + share;
}
}
for (int x = 0; x < SIZE; ++x) {
OMPVV_TEST_AND_SET_VERBOSE(errors, (a[x] - 5 != x));
}
if (num_teams == 1) {
OMPVV_WARNING("Test operated on one team, results of default shared test are inconclusive.");
}
OMPVV_REPORT_AND_RETURN(errors);
}
