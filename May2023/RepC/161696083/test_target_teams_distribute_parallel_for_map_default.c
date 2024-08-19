#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define N 2000
int test_target_teams_distribute_parallel_for_map_default() {
OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_devices");
int a[N];
int b[N];
int c[N];
int d[N];
int scalar = 20;
int scalar2 = -1;
int errors = 0;
int i, j, dev;
for (i = 0; i < N; i++) {
a[i] = 1;
b[i] = i;
c[i] = 2*i;
d[i] = 0;
}
#pragma omp target teams distribute parallel for
for (j = 0; j < N; ++j) {
d[j] += c[j] * (a[j] + b[j] + scalar);
#pragma omp atomic write
scalar2 = j;
} 
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, d[i] != (1 + i + 20) * 2*i);
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_map_default());
OMPVV_REPORT_AND_RETURN(errors);
}
