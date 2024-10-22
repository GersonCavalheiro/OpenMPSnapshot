#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#define SIZE_N 1024
int test_target_teams_distribute_parallel_for_devices() {
OMPVV_INFOMSG("test_target_teams_distribute_parallel_for_devices");
int num_dev = omp_get_num_devices();
int a[SIZE_N];
int isHost[num_dev];
int errors = 0;
int i, dev;
OMPVV_WARNING_IF(num_dev <= 1, "Testing devices clause without multiple devices");
OMPVV_INFOMSG("Num of devices = %d", num_dev);
for (i = 0; i < SIZE_N; i++) {
a[i] = 1;
}
for (dev = 0; dev < num_dev; ++dev) {
#pragma omp target enter data map(to: a[0:SIZE_N]) device(dev)
}
for (dev = 0; dev < num_dev; ++dev) {
#pragma omp target teams distribute parallel for device(dev) map(tofrom: isHost)
for (i = 0; i < SIZE_N; i++) {
if (omp_get_team_num() == 0 && omp_get_thread_num() == 0) {
isHost[dev] = omp_is_initial_device();
}
a[i] += dev;
}
}
for (dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(from: a[0:SIZE_N]) device(dev)
OMPVV_INFOMSG("Device %d ran on the %s", dev, isHost[dev] ? "host" : "device");
OMPVV_TEST_AND_SET(errors, isHost[dev] && dev != omp_get_initial_device());
for (i = 0; i < SIZE_N; i++) {
OMPVV_TEST_AND_SET(errors, a[i] != 1 + dev);
}
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_teams_distribute_parallel_for_devices());
OMPVV_REPORT_AND_RETURN(errors);
}
