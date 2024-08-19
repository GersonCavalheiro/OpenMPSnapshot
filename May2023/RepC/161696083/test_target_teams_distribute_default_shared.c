#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int isOffloading = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
int a[N];
int share = 0;
int errors = 0;
int num_teams;
for (int x = 0; x < N; ++x) {
a[x] = x;
}
#pragma omp target data map(to: a[0:N]) map(tofrom: share, num_teams)
{
#pragma omp target teams distribute default(shared) defaultmap(tofrom:scalar) num_teams(OMPVV_NUM_TEAMS_DEVICE)
for (int x = 0; x < N; ++x) {
if (omp_get_team_num() == 0) {
num_teams = omp_get_num_teams();
}
#pragma omp atomic
share = share + a[x];
}
}
for (int x = 0; x < N; ++x) {
share = share - x;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (share != 0));
share = 5;
#pragma omp target data map(tofrom: a[0:N]) map(tofrom: share)
{
#pragma omp target teams distribute default(shared) defaultmap(tofrom:scalar) num_teams(OMPVV_NUM_TEAMS_DEVICE)
for (int x = 0; x < N; ++x) {
a[x] = a[x] + share;
}
}
for (int x = 0; x < N; ++x) {
OMPVV_TEST_AND_SET_VERBOSE(errors, (a[x] - 5 != x));
if (a[x] - 5 != x) {
break;
}
}
OMPVV_WARNING_IF(num_teams == 1, "Test operated on one team, results of default shared test are inconclusive.");
OMPVV_REPORT_AND_RETURN(errors);
}
