#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define N 1024
int test_target_teams_distribute_parallel_for_num_teams() {
OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_num_teams");
int tested_num_teams[] = {1, 10, 100, 10000};
int num_teams[N]; 
int errors = 0;
int i, nt;
int raiseWarningOneTeam = 0;
for (nt = 0; nt < 4; nt++) {
OMPVV_INFOMSG("Testing for num_teams(%d)", tested_num_teams[nt]);
for (i = 0; i < N; i++) {
num_teams[i] = -1;
}
#pragma omp target teams distribute parallel for          map(tofrom: num_teams) num_teams(tested_num_teams[nt])
for (i = 0; i < N; i++) {
num_teams[i] = omp_get_num_teams();
}
int raiseWarningDifNum = 0;
int prevNumTeams = -1;
for (i = 0; i < N; i++) {
OMPVV_INFOMSG_IF(prevNumTeams != num_teams[i], " %d teams reported", num_teams[i]);
prevNumTeams = num_teams[i];
OMPVV_TEST_AND_SET(errors, num_teams[i] <= 0 || num_teams[i] > tested_num_teams[nt]);
if (num_teams[i] != tested_num_teams[nt]) 
raiseWarningDifNum = 1;
if (num_teams[i] == 1)
raiseWarningOneTeam++; 
}
OMPVV_WARNING_IF(raiseWarningDifNum != 0, "When testing for num_teams(%d), the actual number of teams was different. Not a compliance error with the specs", tested_num_teams[nt]);  
}
OMPVV_WARNING_IF(raiseWarningOneTeam == 4*N, "The num_teams clause always resulted in a single team. Although this is complant with the specs, it is not expected");  
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_num_teams());
OMPVV_REPORT_AND_RETURN(errors);
}
