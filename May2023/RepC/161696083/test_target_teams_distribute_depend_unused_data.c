#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_target_teams_distribute_depend_unused_data() {
int isOffloading = 0;
int errors = 0;
int a[N];
int b[N];
int c[N];
int d[N];
int random_data[1];
for (int x = 0; x < N; ++x) {
a[x] = x;
b[x] = 2 * x;
c[x] = 0;
d[x] = 0;
}
#pragma omp target data map(to: a[0:N], b[0:N]) map(alloc: c[0:N], random_data[0:1]) map(from: d[0:N])
{
#pragma omp target teams distribute nowait depend(out: random_data) map(alloc: a[0:N], b[0:N], c[0:N])
for (int x = 0; x < N; ++x) {
c[x] = a[x] + b[x];
}
#pragma omp target teams distribute nowait depend(out: random_data) map(alloc: b[0:N], c[0:N], d[0:N])
for (int x = 0; x < N; ++x) {
d[x] = c[x] + b[x];
}
#pragma omp taskwait
}
for (int x = 0; x < N; ++x) {
OMPVV_ERROR_IF(d[x] != 5*x, "Test of depend clause using array sections did not pass with offloading %s", (isOffloading ? "enabled" : "disabled"));
OMPVV_TEST_AND_SET_VERBOSE(errors, d[x] != 5*x);
if (d[x] != 5*x) {
break;
}
}
return errors;
}
int main() {
int errors = 0;
int isOffloading = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
errors += test_target_teams_distribute_depend_unused_data();
OMPVV_INFOMSG_IF(errors != 0, "Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
OMPVV_REPORT_AND_RETURN(errors);
}
