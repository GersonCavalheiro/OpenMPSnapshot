#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define N 2000
int test_target_teams_distribute_parallel_for_map_tofrom() {
OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_map_tofrom");
int a[N];
int b[N];
int c[N];
int d[N];
int scalar_to = 50; 
int scalar_from = 50;
int errors = 0;
int i, j, dev;
scalar_to = 50;
scalar_from = 50;
for (i = 0; i < N; i++) {
a[i] = 1;
b[i] = i;
c[i] = 2*i;
d[i] = 0;
}
#pragma omp target teams distribute parallel for map(tofrom: a, b, c, d, scalar_to, scalar_from)
for (j = 0; j < N; ++j) {
d[j] += c[j] * (a[j] + b[j] + scalar_to);
a[j] = 10;
b[j] = 11;
c[j] = 12;
#pragma omp atomic write
scalar_from = 13; 
}
OMPVV_TEST_AND_SET(errors, scalar_from != 13);
for (i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != 10);
OMPVV_TEST_AND_SET(errors, b[i] != 11);
OMPVV_TEST_AND_SET(errors, c[i] != 12);
OMPVV_TEST_AND_SET(errors, d[i] != (1 + i + 50) * 2*i);
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_map_tofrom());
OMPVV_REPORT_AND_RETURN(errors);
}
