#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define N 1024
int interopTestTarget() {
int errors = 0;
int A[N];
int device = omp_get_default_device();
omp_interop_t obj = omp_interop_none;
for (int i = 0; i < N; i++) {
A[i] = 0;
}
#pragma omp interop init(targetsync: obj) device(device) depend(inout: A[0:N]) 
{
#pragma omp target depend(inout: A[0:N]) nowait map(tofrom: A[0:N]) device(device)
for (int j = 0; j < N; j++) {
#pragma omp atomic
A[j] += 5;
}
}
#pragma omp interop destroy(obj) nowait depend(out: A[0:N])
for (int i = 0; i < N; i++) {
OMPVV_TEST_AND_SET(errors, A[i] != 5);
}
return errors;
}
int main () {
int errors = 0;
OMPVV_TEST_OFFLOADING;
int numdevices = omp_get_num_devices();
OMPVV_WARNING_IF(numdevices <= 0, "No devices detected, interop target test running on host");
errors = interopTestTarget();
OMPVV_REPORT_AND_RETURN(errors);
return 0;
}
