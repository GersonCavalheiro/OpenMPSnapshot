#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
int not_shared = 5;
int num_teams = 0; 
#pragma omp target teams default(firstprivate) map(tofrom:num_teams) shared(num_teams) num_teams(OMPVV_NUM_TEAMS_DEVICE)
{
if (omp_get_team_num() == 0) {
num_teams = omp_get_num_teams();
}
for (int i = 0; i < omp_get_num_teams(); i++) {
not_shared += 5;
}
}
OMPVV_WARNING_IF(num_teams != 8, "Number of teams was unexpected, test results likely inconclusive");
OMPVV_TEST_AND_SET(errors, (not_shared != 5));
OMPVV_REPORT_AND_RETURN(errors);
}
