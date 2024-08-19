#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int a[N];
int share = 0;
int errors = 0;
int num_teams;
for (int x = 0; x < N; ++x) {
a[x] = x;
}
#pragma omp teams distribute default(shared) num_teams(OMPVV_NUM_TEAMS_DEVICE)
for ( int i = 0; i < omp_get_num_teams(); i++){
if (omp_get_team_num() == 0) {
num_teams = omp_get_num_teams();
}
#pragma omp atomic
share++;
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (share != num_teams));
OMPVV_WARNING_IF(num_teams == 1, "Test operated on one team, results of default shared test are inconclusive.");
OMPVV_REPORT_AND_RETURN(errors);
}
