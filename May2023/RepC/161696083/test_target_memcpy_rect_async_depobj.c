#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#define N 5
#define M 10
int errors, i, j;
int test_target_memcpy_async_depobj() {
const size_t volume[2] = {5, 10};
const size_t offsets[2] = {0, 0};
const size_t dimensions[2] = {N, M};
int h, t;
errors = 0;
h = omp_get_initial_device();
t = omp_get_default_device();
double hostRect[N][M]; 
double *devRect = (double *)omp_target_alloc(sizeof(double)*N*M, t);
OMPVV_TEST_AND_SET_VERBOSE(errors, devRect == NULL);
for(i = 0; i < N; i++){             
for (j = 0; j < M; j++){
hostRect[i][j] = i + j;
}
}
omp_depend_t obj;
#pragma omp depobj(obj) depend(inout: devRect)
omp_depend_t obj_arr[1] = {obj};
omp_target_memcpy_rect_async(devRect, hostRect, 
sizeof(double), 2, 
volume, 
offsets,          offsets,
dimensions, dimensions,
t,          h,
1,          obj_arr);
#pragma omp taskwait depend(depobj: obj)
#pragma omp target is_device_ptr(devRect) device(t) depend(depobj: obj)
{
for(i = 0; i < N; i++){
for (j = 0; j < M; j++){
devRect[i*M + j] = devRect[i*M + j]*2; 
}
}
}
omp_target_memcpy_rect_async(hostRect, devRect,
sizeof(double), 2,
volume, 
offsets,          offsets,
dimensions, dimensions,
h,          t,
1,          obj_arr);
#pragma omp taskwait depend(depobj: obj)
for(i = 0; i < N; i++){
for(j = 0; j < N; j++){
OMPVV_TEST_AND_SET(errors, hostRect[i][j]!=(i+j)*2);
}
}
omp_target_free(devRect, t);
#pragma omp depobj(obj) destroy
return errors;
}
int main() {
errors = 0;
OMPVV_TEST_OFFLOADING;
OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_memcpy_async_depobj() != 0);
OMPVV_REPORT_AND_RETURN(errors);
}
