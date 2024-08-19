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
int sum = 0;
int errors = 0;
int isHost[num_dev];
int h_matrix[N];
for (int dev = 0; dev < num_dev; ++dev) {
omp_set_default_device(dev);
{
#pragma omp target enter data map(alloc: h_matrix[0:N])
printf(""); 
}
}
for (int i = 0; i < N; ++i) {
h_matrix[i] = 0;
}
for (int dev = 0; dev < num_dev; ++dev) {
omp_set_default_device(dev);
#pragma omp target update to(h_matrix[0:N])
#pragma omp target map(alloc: h_matrix[0:N]) map(tofrom: isHost[dev:1]) 
{
isHost[dev] = omp_is_initial_device();
for (int i = 0; i < N; ++i) {
h_matrix[i]++;
}
}
#pragma omp target update from(h_matrix[0:N])
}
for (int dev = 0; dev < num_dev; ++dev) {
omp_set_default_device(dev);
#pragma omp target exit data map(delete: h_matrix[0:N])
printf("");
}
for (int dev = 0; dev < num_dev; ++dev) {
OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
}
for (int i = 0; i < N; ++i) {
sum += h_matrix[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (num_dev * N != sum));
omp_set_default_device(def_dev);
return errors;
}
int test_device() {
OMPVV_INFOMSG("test_device_clause");
int num_dev = omp_get_num_devices();
OMPVV_INFOMSG("num_devices: %d", num_dev);
int def_dev = omp_get_default_device();
OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
OMPVV_INFOMSG("default device: %d", def_dev);
int sum = 0;
int errors = 0;
int isHost[num_dev];
int h_matrix[N];
for (int dev = 0; dev < num_dev; ++dev) {
{
#pragma omp target enter data map(alloc: h_matrix[0:N]) device(dev)
printf(""); 
}
}
for (int i = 0; i < N; ++i) {
h_matrix[i] = 0;
}
for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target update to(h_matrix[0:N]) device(dev)
#pragma omp target map(alloc: h_matrix[0:N]) map(tofrom: isHost[dev:1]) device(dev)
{
isHost[dev] = omp_is_initial_device();
for (int i = 0; i < N; ++i) {
h_matrix[i]++;
}
}
#pragma omp target update from(h_matrix[0:N]) device(dev)
}
for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(delete: h_matrix[0:N]) device(dev)
printf("");
}
for (int dev = 0; dev < num_dev; ++dev) {
OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
}
for (int i = 0; i < N; ++i) {
sum += h_matrix[i];
}
OMPVV_TEST_AND_SET_VERBOSE(errors, (num_dev * N != sum));
return errors;
}
int main() {
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET(errors, test_set_default_dev());
OMPVV_TEST_AND_SET(errors, test_device());
OMPVV_REPORT_AND_RETURN(errors);
}
