#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define SIZE 1024
#define SIZE_THRESHOLD 512
int test_target_data_map_if_nested (int isOffloading){
int a[SIZE];
int b[SIZE];
int c[SIZE];
int map_size, i = 0, errors[2] = {0,0}, isHost = -1;
for (i = 0; i < SIZE; i++) {
a[i] = 1;
b[i] = i;
}
for (map_size = 256; map_size <= SIZE; map_size += 256) {
for (i = 0; i < map_size; i++) {
c[i] = -1;
}
#pragma omp target data if(map_size > SIZE_THRESHOLD) map(to: map_size)  map(tofrom: c[0:map_size])                                       map(to: a[0:map_size], b[0:map_size])
{
#pragma omp target if(map_size > SIZE_THRESHOLD) map(tofrom: isHost) map (alloc: a[0:map_size], b[0:map_size], c[0:map_size]) 
{
isHost = omp_is_initial_device();
int alpha = (isHost ? 0 : 1);
int j = 0;
for (j = 0; j < map_size; j++) {
c[j] = alpha*(a[j] + b[j]);
}
} 
}
for (i = 0; i < map_size; i++) {
if (isOffloading && map_size > SIZE_THRESHOLD) {
OMPVV_TEST_AND_SET(errors[0], (c[i] != i + 1)); 
} else {
OMPVV_TEST_AND_SET(errors[1], (c[i] != 0));
} 
}
} 
if (!errors[0] && !errors[1]) {
OMPVV_INFOMSG("Test nested if passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
} else if (errors[0]==0 && errors[1]!=0) {
OMPVV_ERROR("Test nested if failed on host with offloading %s.", (isOffloading ? "enabled" : "disabled"));
} else if (errors[0]!=0 && errors[1]==0) {
OMPVV_ERROR("Test nested if failed on device with offloading %s.", (isOffloading ? "enabled" : "disabled"));
} else if (errors[0]!=0 && errors[1]!=0) {
OMPVV_ERROR("Test nested if failed on host and device with offloading %s.", (isOffloading ? "enabled" : "disabled"));
}
return errors[0] + errors[1];
}
int test_target_data_map_if_simple(int isOffloading){
int a[SIZE];
int b[SIZE];
int c[SIZE];
int map_size, i = 0, errors[3] = {0,0,0}, isHost = -1;
for (map_size = 256; map_size <= SIZE; map_size += 256) {
for (i = 0; i < SIZE; i++) {
a[i] = SIZE - i;
b[i] = i;
c[i] = -1;
}
#pragma omp target data if(map_size > SIZE_THRESHOLD) map(to: map_size)  map(tofrom: c[0:map_size])                             map(to: a[0:map_size], b[0:map_size])
{
#pragma omp target map(tofrom: isHost) map (alloc: a[0:map_size], b[0:map_size], c[0:map_size]) 
{
isHost = omp_is_initial_device();
int j = 0;
for (j = 0; j < map_size; j++) {
c[j] += (a[j] + b[j] + 1);
a[j] = -1; 
b[j] = -1; 
}
} 
if (isOffloading) {
OMPVV_TEST_AND_SET_VERBOSE(errors[0], isHost);
}
}
for (i = 0; i < map_size; i++) {
if (map_size > SIZE_THRESHOLD || !isOffloading) {
OMPVV_TEST_AND_SET(errors[1], (c[i] != SIZE)); 
} else {
OMPVV_TEST_AND_SET(errors[2], (c[i] != -1));
} 
}
} 
if (errors[0]) {
OMPVV_ERROR("Test did not offload to the device. 'If' clause might be affecting the target"
" offlading as well and it should not ")
}
if (!errors[0] && !errors[1] && !errors[2]) {
OMPVV_INFOMSG("Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
} else if (errors[1]==0 && errors[2]!=0) {
OMPVV_ERROR("Test failed for if (false) with offloading %s.", (isOffloading ? "enabled" : "disabled"));
} else if (errors[1]!=0 && errors[2]==0) {
OMPVV_ERROR("Test failed for if (true) with offloading %s.", (isOffloading ? "enabled" : "disabled"));
} else if (errors[1]!=0 && errors[2]!=0) {
OMPVV_ERROR("Test failed for if(true) and if(false) with offloading %s.", (isOffloading ? "enabled" : "disabled"));
}
return errors[0] + errors[1] + errors[2];
}
int main() {
int isOffloading = 0;
int errors = 0;
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "Offloading is off, tests will be inconclusive. No way to tests if");
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_data_map_if_nested(isOffloading) != 0);
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_data_map_if_simple(isOffloading) != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
