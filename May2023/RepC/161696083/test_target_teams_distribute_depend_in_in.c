#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_target_teams_distribute_depend_in_in() {
int isOffloading = 0;
int a[N];
int b[N];
int c[N];
int d[N];
int async_found = 0;
int errors = 0;
for (int x = 0; x < N; ++x) {
a[x] = x;
b[x] = 2*x;
c[x] = 0;
d[x] = 0;
}
#pragma omp target data map(to: a[0:N], b[0:N]) map(tofrom: c[0:N], d[0:N])
{
#pragma omp target teams distribute nowait depend(in:d) map(alloc: a[0:N], b[0:N], d[0:N])
for (int x = 0; x < N; ++x) {
#pragma omp atomic
d[x] += a[x] + b[x];
}
#pragma omp target teams distribute nowait depend(in:d) map(alloc: a[0:N], b[0:N], c[0:N], d[0:N])
for (int x = 0; x < N; ++x) {
#pragma omp atomic
c[x] += 2*(a[x] + b[x]) + d[x];
}
#pragma omp taskwait
}
for (int x = 0; x < N; ++x) {
OMPVV_TEST_AND_SET_VERBOSE(errors, c[x] != 6*x && c[x] != 9*x);
OMPVV_ERROR_IF(errors, "Found invalid results, cannot show independence between depend clauses on disjoint array sections.");
if (errors) {
break;
}
if (c[x] == 6*x) {
async_found = 1;
}
}
OMPVV_INFOMSG_IF(!errors && async_found, "Found asynchronous behavior between depend clauses on disjoint array sections.");
OMPVV_WARNING_IF(!errors && !async_found, "Constructs ran in sequence, could not show lack of dependence since nowait had no effect.");
return errors;
}
int main() {
int errors = 0;
int isOffloading = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
errors += test_target_teams_distribute_depend_in_in();
OMPVV_INFOMSG_IF(errors != 0, "Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
OMPVV_REPORT_AND_RETURN(errors);
}
