#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int main() {
int isOffloading = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
int a[N];
int b[N];
int c[N];
int num_teams[N];
int default_num_teams = 0;
int errors = 0;
for (int x = 0; x < N; ++x) {
a[x] = 1;
b[x] = x;
c[x] = 0;
}
#pragma omp target teams distribute map(tofrom: default_num_teams, c[0:N]) map(to: a[0:N], b[0:N])
for (int x = 0; x < N; ++x) {
if (omp_get_team_num() == 0) {
default_num_teams = omp_get_num_teams();
}
c[x] = a[x] + b[x];
}
if (default_num_teams == 1) {
OMPVV_WARNING("Test operated with one team.  Testing num_teams clause cannot be done.");
} else if(default_num_teams <= 0) {
OMPVV_ERROR("Test returned num_teams <= 0.");
errors = 1;
} else {
#pragma omp target teams distribute num_teams(default_num_teams / 2) map(to: a[0:N], b[0:N]) map(from: c[0:N], num_teams[0:N])
for (int x = 0; x < N; ++x) {
num_teams[x] = omp_get_num_teams();
c[x] = a[x] + b[x];
}
for (int i = 1; i < N; ++i) {
if (num_teams[i] != num_teams[i - 1]) {
errors += 1;
OMPVV_ERROR("omp_get_num_teams returned an inconsistent number of teams between iterations.");
break;
}
}
if (num_teams[0] > default_num_teams / 2) {
errors += 1;
OMPVV_ERROR("Test ran on more teams than requested.");
return errors;
} else if (num_teams[0] < default_num_teams / 2) {
OMPVV_WARNING("Test ran on less teams than requested. This is still spec-conformant.");
} else {
OMPVV_INFOMSG("Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
}
}
OMPVV_REPORT_AND_RETURN(errors);
}
