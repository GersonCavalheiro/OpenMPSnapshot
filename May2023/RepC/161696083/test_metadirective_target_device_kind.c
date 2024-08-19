#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int test_metadirective_target_device() {
int errors = 0;
int A[N];
for (int i = 0; i < N; i++) {
A[i] = 1;
}
#pragma omp metadirective when( target_device={kind(gpu)}: target defaultmap(none) map(tofrom: A)) when( target_device={kind(nohost)}: target defaultmap(none) map(tofrom: A)) default( target defaultmap(none) map(to: A))
for(int i = 0; i < N; i++){
A[i] = i;
}
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, A[i] != i);
}
OMPVV_INFOMSG("Test ran with a number of available devices greater than 0");
return errors;
}
int main () {
int errors = 0;
OMPVV_TEST_OFFLOADING;
if (omp_get_num_devices() > 0) {
errors = test_metadirective_target_device();
} else {
OMPVV_INFOMSG("Cannot test target_device without a target device!")
}
OMPVV_REPORT_AND_RETURN(errors);
return 0;
}
