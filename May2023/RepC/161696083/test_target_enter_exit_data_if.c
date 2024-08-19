#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define SIZE_THRESHOLD 512
int main() {
int a[1024];
int b[1024];
int c[1024];
int size, i = 0, errors[2] = {0,0}, isOffloading = 0;
for (i = 0; i < 1024; i++) {
a[i] = 1;
b[i] = i;
}
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
OMPVV_WARNING_IF(!isOffloading, "It is not possible to test conditional data transfers "
"if the environment is shared or offloading is off. Not testing "
"anything"); 
for (size = 256; size <= 1024; size += 256) {
for (i = 0; i < size; i++) {
c[i] = -1;
}
#pragma omp target enter data if(size > SIZE_THRESHOLD) map(to: size) map(to: c[0:size])
#pragma omp target if(size > SIZE_THRESHOLD)  map(to: a[0:size], b[0:size]) map(to: c[0:size]) 
{
int isHost = -1;
isHost = omp_is_initial_device();
int alpha = (isHost ? 0 : 1);
int j = 0;
for (j = 0; j < size; j++) {
c[j] = alpha*(a[j] + b[j]);
}
} 
#pragma omp target exit data if(size > SIZE_THRESHOLD) map(from: c[0:size])
for (i = 0; i < size; i++) {
if (isOffloading && size > SIZE_THRESHOLD) {
if (c[i] != i + 1) {
errors[0] += 1;
}
} else {
if (c[i] != 0) {
errors[1] += 1;
}
} 
}
} 
if (!errors[0] && !errors[1]) {
OMPVV_INFOMSG("Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
} else if (errors[0]==0 && errors[1]!=0) {
OMPVV_ERROR("Test failed on host with offloading %s.", (isOffloading ? "enabled" : "disabled"));
} else if (errors[0]!=0 && errors[1]==0) {
OMPVV_ERROR("Test failed on device with offloading %s.", (isOffloading ? "enabled" : "disabled"));
} else if (errors[0]!=0 && errors[1]!=0) {
OMPVV_ERROR("Test failed on host and device with offloading %s.", (isOffloading ? "enabled" : "disabled"));
}
OMPVV_REPORT_AND_RETURN((errors[0] && errors[1]));
}
