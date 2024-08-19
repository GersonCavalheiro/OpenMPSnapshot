#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h" 
#define SIZE_THRESHOLD 512
int main() {
int isOffloading = 0; 
OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
if (!isOffloading)
OMPVV_WARNING("It is not possible to test conditional data transfers "
"if the environment is shared or offloading is off. Not testing "
"anything")
int a[1024];
int b[1024];
int c[1024];
int size, i = 0, errors[2] = {0,0}, isHost = -1;
for (i = 0; i < 1024; i++) {
a[i] = 1;
b[i] = i;
}
for (size = 256; size <= 1024; size += 256) {
for (i = 0; i < size; i++) {
c[i] = -1;
}
#pragma omp target if(size > SIZE_THRESHOLD) map(to: size)  map(tofrom: c[0:size])                                       map(to: a[0:size], b[0:size])  map(tofrom: isHost)
{
isHost = omp_is_initial_device();
int alpha = (isHost ? 0 : 1);
int j = 0;
for (j = 0; j < size; j++) {
c[j] = alpha*(a[j] + b[j]);
}
} 
for (i = 0; i < size; i++) {
if (isOffloading && size > SIZE_THRESHOLD) {
OMPVV_TEST_AND_SET(errors[0], (c[i] != i + 1));
} else {
OMPVV_TEST_AND_SET(errors[1], (c[i] != 0));
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
OMPVV_REPORT_AND_RETURN((errors[0] + errors[1]));
}
