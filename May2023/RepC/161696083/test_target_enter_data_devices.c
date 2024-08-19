#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1000
int test_set_default_dev() {
OMPVV_INFOMSG("test_set_default_dev");
int num_dev = omp_get_num_devices();
OMPVV_INFOMSG("num_devices: %d", num_dev);
int def_dev = omp_get_default_device();
OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
OMPVV_INFOMSG("default device: %d", def_dev);
int sum[num_dev], errors = 0, isHost[num_dev];
int h_matrix[num_dev][N], h_matrix_copy[num_dev][N];
for (int dev = 0; dev < num_dev; ++dev) {
sum[dev] = 0;
isHost[num_dev] = 0;
}
for (int dev = 0; dev < num_dev; ++dev) {
omp_set_default_device(dev);
{
#pragma omp target enter data map(alloc: h_matrix[dev][0:N]) 
printf(""); 
}
#pragma omp target map(alloc: h_matrix[dev][0:N]) map(tofrom: isHost[dev:1]) 
{
isHost[dev] = omp_is_initial_device();
for (int i = 0; i < N; ++i) {
h_matrix[dev][i] = dev;
}
}
#pragma omp target map(from: h_matrix_copy[dev][0:N]) map(alloc: h_matrix[dev][0:N])
{
for (int i = 0; i < N; ++i) {
h_matrix_copy[dev][i] = h_matrix[dev][i];
}
}
}
for (int dev = 0; dev < num_dev; ++dev) {
OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
for (int i = 0; i < N; ++i)
sum[dev] += h_matrix_copy[dev][i];
OMPVV_TEST_AND_SET(errors, (dev * N != sum[dev]));
}
omp_set_default_device(def_dev);
for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(delete: h_matrix[dev][0:N]) device(dev)
}
return errors;
}
int test_device() {
OMPVV_INFOMSG("test_device");
int num_dev = omp_get_num_devices();
OMPVV_INFOMSG("num_devices: %d", num_dev);
OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
OMPVV_INFOMSG("default device: %d", omp_get_default_device());
int sum[num_dev], errors = 0, isHost[num_dev];
int h_matrix[num_dev][N], h_matrix_copy[num_dev][N];
for (int dev = 0; dev < num_dev; ++dev) {
sum[dev] = 0;
isHost[num_dev] = 0;
}
for (int dev = 0; dev < num_dev; ++dev) {
{
#pragma omp target enter data map(alloc: h_matrix[dev][0:N]) device(dev)
printf("");
}
#pragma omp target map(alloc: h_matrix[dev][0:N]) map(tofrom: isHost[dev:1]) device(dev) 
{
isHost[dev] = omp_is_initial_device();
for (int i = 0; i < N; ++i)
h_matrix[dev][i] = dev;
}
#pragma omp target map(from: h_matrix_copy[dev][0:N])  map(alloc: h_matrix[dev][0:N]) device(dev)
{
for (int i = 0; i < N; ++i) {
h_matrix_copy[dev][i] = h_matrix[dev][i];
}
}
}
for (int dev = 0; dev < num_dev; ++dev) {
OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
for (int i = 0; i < N; ++i)
sum[dev] += h_matrix_copy[dev][i];
OMPVV_TEST_AND_SET(errors, (dev * N != sum[dev]));
}
for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(delete: h_matrix[dev][0:N]) device(dev)
}
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET(errors, test_set_default_dev());
OMPVV_TEST_AND_SET(errors, test_device());
OMPVV_REPORT_AND_RETURN(errors);
}
