#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ompvv.h"
#define N 1000
#define HOST_TASK1_BIT 0x1
#define HOST_TASK2_BIT 0x2
#define DEVICE_TASK1_BIT 0x4
#define DEVICE_TASK2_BIT 0x8
#define HOST_TASK3_BIT 0x10
#define ALL_TASKS_BITS 0x1F
int test_async_between_task_target() {
OMPVV_INFOMSG("test_async_between_task_target");
int errors = 0;
bool isHost = true;
int sum = 0.0;
int* h_array = (int *) malloc(N * sizeof(int));
int* h_array_copy = (int *) malloc(N * sizeof(int));
int* in_1 = (int *) malloc(N * sizeof(int));
int* in_2 = (int *) malloc(N * sizeof(int));
#pragma omp task depend(out: in_1) shared(in_1)
{
for (int i = 0; i < N; ++i) {
in_1[i] = HOST_TASK1_BIT; 
}
}
#pragma omp task depend(out: in_2) shared(in_2)
{
for (int i = 0; i < N; ++i) {
in_2[i] = HOST_TASK2_BIT; 
}
}
#pragma omp target enter data map(alloc: h_array[0:N]) map(to: in_1[0:N]) map(to: in_2[0:N]) depend(out: h_array) depend(in: in_1) depend(in: in_2) 
#pragma omp task shared (isHost, h_array, in_1, in_2) depend(inout: h_array) depend(in: in_1) depend(in: in_2)
{
#pragma omp target map(tofrom: isHost) map(alloc: in_1[0:N]) map(alloc: in_2[0:N]) map(alloc: h_array[0:N])
{
isHost = omp_is_initial_device();
for (int i = 0; i < N; ++i) {
h_array[i] = DEVICE_TASK1_BIT | in_1[i] | in_2[i]; 
}
}
}
#pragma omp task shared (h_array, h_array_copy) depend(in: h_array) depend(out: h_array_copy)
{
#pragma omp target map(alloc: h_array[0:N]) map(from: h_array_copy[0:N])
{
for (int i = 0; i < N; ++i) {
h_array_copy[i] = h_array[i] | DEVICE_TASK2_BIT; 
}
}
}
#pragma omp task depend(in: h_array_copy) shared(sum, h_array_copy)
{
for (int i = 0; i < N; ++i) {
h_array_copy[i] |= HOST_TASK3_BIT;
sum += (h_array_copy[i] & ALL_TASKS_BITS); 
}
}
#pragma omp taskwait
int h_task1 = 0;
int h_task2 = 0;
int h_task3 = 0;
int d_task1 = 0;
int d_task2 = 0;
for (int i = 0; i < N; ++i) {
h_task1 |= !(h_array_copy[i] & HOST_TASK1_BIT);
h_task2 |= !(h_array_copy[i] & HOST_TASK2_BIT);
h_task2 |= !(h_array_copy[i] & HOST_TASK3_BIT);
d_task1 |= !(h_array_copy[i] & DEVICE_TASK1_BIT);
d_task2 |= !(h_array_copy[i] & DEVICE_TASK2_BIT);
}
OMPVV_ERROR_IF(h_task1 != 0, "Error in host task 1");
OMPVV_ERROR_IF(h_task2 != 0, "Error in host task 2");
OMPVV_ERROR_IF(h_task3 != 0, "Error in host task 3");
OMPVV_ERROR_IF(d_task1 != 0, "Error in device task 1");
OMPVV_ERROR_IF(d_task2 != 0, "Error in device task 2");
OMPVV_TEST_AND_SET(errors, (N * ALL_TASKS_BITS != sum));
OMPVV_INFOMSG("Test test_async_between_task_target ran on the %s", (isHost ? "host" : "device"));
#pragma omp target exit data map(delete: h_array[0:N], in_1[0:N], in_2[0:N])
free(h_array);
free(h_array_copy);
free(in_1);
free(in_2);
return errors;
}
int test_async_between_target() {
OMPVV_INFOMSG("test_async_between_target");
int errors = 0;
bool isHost = true;
int sum = 0;
int* h_array = (int *) malloc(N * sizeof(int));
int* h_array_copy = (int *) malloc(N * sizeof(int));
int val = DEVICE_TASK1_BIT;
#pragma omp target enter data map(alloc: h_array[0:N]) depend(out: h_array) 
#pragma omp target enter data map(to: val) depend(out: val) 
#pragma omp target map(tofrom: isHost) map(alloc: h_array[0:N]) depend(inout: h_array) depend(in: val) 
{
isHost = omp_is_initial_device();
for (int i = 0; i < N; ++i) {
h_array[i] = val; 
}
}
#pragma omp target map(alloc: h_array[0:N]) map(from: h_array_copy[0:N]) depend(in: h_array) depend(out: h_array_copy) 
{
for (int i = 0; i < N; ++i) {
h_array_copy[i] = h_array[i] | DEVICE_TASK2_BIT;
}
}
#pragma omp taskwait
int d_task1 = 0;
int d_task2 = 0;
for (int i = 0; i < N; ++i) {
sum += (h_array_copy[i] & (DEVICE_TASK1_BIT | DEVICE_TASK2_BIT)); 
d_task1 |= !(h_array_copy[i] & DEVICE_TASK1_BIT);
d_task2 |= !(h_array_copy[i] & DEVICE_TASK2_BIT);
}
OMPVV_ERROR_IF(d_task1 != 0, "Error in device task 1");
OMPVV_ERROR_IF(d_task2 != 0, "Error in device task 2");
OMPVV_TEST_AND_SET(errors, (N * (DEVICE_TASK1_BIT | DEVICE_TASK2_BIT) != sum));
OMPVV_INFOMSG("Test test_async_between_task_target ran on the %s", (isHost ? "host" : "device"));
#pragma omp target exit data map(delete: h_array[0:N], val)
free(h_array);
free(h_array_copy);
return errors;
}
int main(){
OMPVV_TEST_OFFLOADING;
int errors = 0;
OMPVV_TEST_AND_SET(errors, test_async_between_target());
OMPVV_TEST_AND_SET(errors, test_async_between_task_target());
OMPVV_REPORT_AND_RETURN(errors);
}
