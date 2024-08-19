#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "metamorph.h"
#ifndef WITH_INTELFPGA 
#define KERNEL_REDUCE
#define KERNEL_STENCIL
#define KERNEL_DOT_PROD
#define KERNEL_TRANSPOSE
#endif 
#if (!defined(KERNEL_REDUCE) && !defined(KERNEL_STENCIL) && !defined(KERNEL_DOT_PROD) && !defined(KERNEL_TRANSPOSE))
#error Define at least one of the function  among KERNEL_STENCIL, KERNEL_DOT_PROD, KERNEL_TRANSPOSE, KERNEL_REDUCE
#endif 
meta_type_id g_type;
size_t g_typesize;
void set_type(meta_type_id type) {
char data_type[30];
switch (type) {
case meta_db:
g_typesize = sizeof(double);
strcpy(data_type, "Double");
break;
case meta_fl:
g_typesize = sizeof(float);
strcpy(data_type, "Float");
break;
case meta_ul:
g_typesize = sizeof(unsigned long);
strcpy(data_type, "Unsigned Long");
break;
case meta_in:
g_typesize = sizeof(int);
strcpy(data_type, "Integer");
break;
case meta_ui:
g_typesize = sizeof(unsigned int);
strcpy(data_type, "Unsigned Integer");
break;
default:
fprintf(stderr, "Unsupported type: [%d] provided to set_type!\n", type);
exit(-1);
break;
}
printf("-INFO- Data Type : %0s\n-INFO- Data Size : %0d\n", data_type,
g_typesize);
g_type = type;
}
void * dev_data3, *dev_data3_2, *reduction;
#ifdef DATA4
void *dev_data4;
#endif
void *data3;
#ifdef DATA4
void *data4;
#endif
int ni, nj, nk, nm, iters;
void data_allocate(int i, int j, int k, int m) {
a_err istat = 0;
data3 = malloc(g_typesize * ni * nj * nk);
#ifdef DATA4
data4 = malloc(g_typesize*ni*nj*nk*nm);
#endif
#ifndef UNIFIED_MEMORY
istat = meta_alloc(&dev_data3, g_typesize * ni * nj * nk);
#endif
istat = meta_alloc(&dev_data3_2, g_typesize * ni * nj * nk);
#ifdef DATA4
istat = meta_alloc( &dev_data4, g_typesize*ni*nj*nk*nm);
#endif
istat = meta_alloc(&reduction, g_typesize);
}
void data_initialize() {
int i, j, k;
switch (g_type) {
default:
case meta_db: {
double * l_data3 = (double *) data3;
#ifdef DATA4
double * l_data4 = (double *) data4;
#endif
for (i = ni * nj * nk * nm - 1; i >= ni * nj * nk; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
}
for (; i >= 0; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
l_data3[i] = 1;
}
k = 0;
for (; k < nk; k++) {
j = 0;
for (; j < nj; j++) {
i = 0;
for (; i < ni; i++) {
if (i == 0 || j == 0 || k == 0)
l_data3[i + j * ni + k * ni * nj] = 0;
if (i == ni - 1 || j == nj - 1 || k == nk - 1)
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
break;
case meta_fl: {
float * l_data3 = (float *) data3;
#ifdef DATA4
float * l_data4 = (float *) data4;
#endif
for (i = ni * nj * nk * nm - 1; i >= ni * nj * nk; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
}
for (; i >= 0; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
l_data3[i] = 1;
}
k = 0;
for (; k < nk; k++) {
j = 0;
for (; j < nj; j++) {
i = 0;
for (; i < ni; i++) {
if (i == 0 || j == 0 || k == 0)
l_data3[i + j * ni + k * ni * nj] = 0;
if (i == ni - 1 || j == nj - 1 || k == nk - 1)
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
break;
case meta_ul: {
unsigned long * l_data3 = (unsigned long *) data3;
#ifdef DATA4
unsigned long * l_data4 = (unsigned long *) data4;
#endif
for (i = ni * nj * nk * nm - 1; i >= ni * nj * nk; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
}
for (; i >= 0; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
l_data3[i] = 1;
}
k = 0;
for (; k < nk; k++) {
j = 0;
for (; j < nj; j++) {
i = 0;
for (; i < ni; i++) {
if (i == 0 || j == 0 || k == 0)
l_data3[i + j * ni + k * ni * nj] = 0;
if (i == ni - 1 || j == nj - 1 || k == nk - 1)
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
break;
case meta_in: {
int * l_data3 = (int *) data3;
#ifdef DATA4
int * l_data4 = (int *) data4;
#endif
for (i = ni * nj * nk * nm - 1; i >= ni * nj * nk; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
}
for (; i >= 0; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
l_data3[i] = 1;
}
k = 0;
for (; k < nk; k++) {
j = 0;
for (; j < nj; j++) {
i = 0;
for (; i < ni; i++) {
if (i == 0 || j == 0 || k == 0)
l_data3[i + j * ni + k * ni * nj] = 0;
if (i == ni - 1 || j == nj - 1 || k == nk - 1)
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
break;
case meta_ui: {
unsigned int * l_data3 = (unsigned int *) data3;
#ifdef DATA4
unsigned int * l_data4 = (unsigned int *) data4;
#endif
for (i = ni * nj * nk * nm - 1; i >= ni * nj * nk; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
}
for (; i >= 0; i--) {
#ifdef DATA4
l_data4[i] = 1;
#endif
l_data3[i] = 1;
}
k = 0;
for (; k < nk; k++) {
j = 0;
for (; j < nj; j++) {
i = 0;
for (; i < ni; i++) {
if (i == 0 || j == 0 || k == 0)
l_data3[i + j * ni + k * ni * nj] = 0;
if (i == ni - 1 || j == nj - 1 || k == nk - 1)
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
break;
}
}
void data_transfer_h2d() {
a_err ret;
#ifdef WITH_CUDA
ret= cudaSuccess;
#endif
#ifdef WITH_OPENCL
ret= CL_SUCCESS;
#endif
#ifdef WITH_OPENMP
ret= 0;
#endif
int iter;
struct timeval start, end;
#ifndef UNIFIED_MEMORY
gettimeofday(&start, NULL);
for (iter = 0; iter < iters; iter++)
ret |= meta_copy_h2d(dev_data3, data3, g_typesize * ni * nj * nk,
false);
gettimeofday(&end, NULL);
printf("D2H time: %f\n",
((end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec)) / (iters));
#else
dev_data3 = data3;
#endif
#ifdef DATA4
ret |= meta_copy_h2d( dev_data4, data4, g_typesize*ni*nj*nk*nm, false);
#endif
gettimeofday(&start, NULL);
for (iter = 0; iter < iters; iter++)
ret |= meta_copy_d2d(dev_data3_2, dev_data3, g_typesize * ni * nj * nk,
false);
gettimeofday(&end, NULL);
printf("D2D time: %f\n",
((end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec)) / (iters));
}
void deallocate_() {
#ifndef UNIFIED_MEMORY
meta_free(dev_data3);
#endif
free(data3);
meta_free(dev_data3_2);
#ifdef DATA4
meta_free(dev_data4);
free(data4);
#endif
meta_free(reduction);
}
void gpu_initialize() {
int istat, deviceused; 
int idevice;
#ifdef WITH_CUDA
idevice = 0;
istat = meta_set_acc(idevice, metaModePreferCUDA); 
#endif
#ifdef WITH_OPENCL
idevice = -1;
istat = meta_set_acc(idevice, metaModePreferGeneric); 
#endif
#ifdef WITH_OPENMP
idevice = 0;
istat = meta_set_acc(idevice, metaModePreferOpenMP); 
#endif
meta_preferred_mode mode;
istat = meta_get_acc(&deviceused, &mode); 
} 
void print_grid(double * grid) {
int i, j, k;
for (k = 0; k < nk; k++) {
for (j = 0; j < nj; j++) {
for (i = 0; i < ni; i++) {
printf("[%f] ", grid[i + j * (ni) + k * nj * (ni)]);
}
printf("\n");
}
printf("\n");
}
}
int main(int argc, char **argv) {
int tx, ty, tz, gx, gy, gz, istat, i, l_type;
meta_dim3 dimgrid, dimblock, dimarray, arr_start, arr_end; 
meta_dim3 trans_2d;
char args[32];
i = argc;
if (i != 10) {
printf(
"<ni> <nj> <nk> <nm> <tblockx> <tblocky> <tblockz> <type> <iters>");
return (1); 
}
ni = atoi(argv[1]);
nj = atoi(argv[2]);
nk = atoi(argv[3]);
nm = atoi(argv[4]);
tx = atoi(argv[5]);
ty = atoi(argv[6]);
tz = atoi(argv[7]);
l_type = atoi(argv[8]); 
set_type((meta_type_id) l_type);
iters = atoi(argv[9]);
void * sum_dot_gpu, *zero;
sum_dot_gpu = malloc(g_typesize);
zero = malloc(g_typesize);
#ifdef WITH_TIMERS
metaTimersInit();
#endif
gpu_initialize();           
data_allocate(ni, nj, nk, nm); 
data_initialize();          
data_transfer_h2d(); 
printf("Performing dot-product, type %d\n", l_type); 
#ifdef WITH_OPENMP
#pragma omp parallel
{
#pragma omp master
printf("num threads %d\n", omp_get_num_threads());
}
#endif
dimblock[0] = tx, dimblock[1] = ty, dimblock[2] = tz;
if ((nj) % ty != 0)  
gy = (nj) / ty + 1;
else
gy = (nj) / ty;
if ((ni) % tx != 0) 
gx = (ni) / tx + 1;
else
gx = (ni) / tx;
if ((nk) % tz != 0) 
gz = (nk) / tz + 1;
else
gz = (nk) / tz;
dimgrid[0] = gx, dimgrid[1] = gy, dimgrid[2] = gz;
switch (g_type) {
case meta_db:
*(double*) zero = 0;
break;
case meta_fl:
*(float*) zero = 0;
break;
case meta_ul:
*(unsigned long*) zero = 0;
break;
case meta_in:
*(int *) zero = 0;
break;
case meta_ui:
*(unsigned int *) zero = 0;
break;
}
dimarray[0] = ni, dimarray[1] = nj, dimarray[2] = nk;
arr_start[0] = arr_start[1] = arr_start[2] = 0;
arr_end[0] = ni - 1, arr_end[1] = nj - 1, arr_end[2] = nk - 1;
trans_2d[0] = ni;
trans_2d[1] = nj * nk;
trans_2d[2] = 1;
istat = meta_copy_h2d(reduction, zero, g_typesize, true);
for (; meta_validate_worksize(&dimgrid, &dimblock) != 0 && dimblock[2] > 1;
dimgrid[2] <<= 1, dimblock[2] >>= 1)
;
if (meta_validate_worksize(&dimgrid, &dimblock)) {
}
int iter;
struct timeval start, end;
a_err ret;
#ifdef KERNEL_DOT_PROD
printf("DOT product\n");
gettimeofday(&start, NULL);
for (iter = 0; iter < iters; iter++)
ret = meta_dotProd(&dimgrid, &dimblock, dev_data3, dev_data3_2,
&dimarray, &arr_start, &arr_end, reduction, g_type, false);
gettimeofday(&end, NULL);
fprintf(stderr, "Kernel Status: %d\n", ret);
printf("Kern time: %f\n",
((end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec)) / (iters));
istat = meta_copy_d2h(sum_dot_gpu, reduction, g_typesize, false);
switch (g_type) {
case meta_db:
printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%f]\n",
(*(double*) sum_dot_gpu
== (double) ((ni - 2) * (nj - 2) * (nk - 2) * iters) ?
"PASSED" : "FAILED"),
(ni - 2) * (nj - 2) * (nk - 2) * iters,
(*(double*) sum_dot_gpu)); 
break;
case meta_fl:
printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%f]\n",
(*(float*) sum_dot_gpu
== (float) ((ni - 2) * (nj - 2) * (nk - 2) * iters) ?
"PASSED" : "FAILED"),
(ni - 2) * (nj - 2) * (nk - 2) * iters,
(*(float*) sum_dot_gpu)); 
break;
case meta_ul:
printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%ld]\n",
(*(unsigned long*) sum_dot_gpu
== (unsigned long) ((ni - 2) * (nj - 2) * (nk - 2)
* iters) ? "PASSED" : "FAILED"),
(ni - 2) * (nj - 2) * (nk - 2) * iters,
(*(unsigned long*) sum_dot_gpu)); 
printf("Test Dot-Product:\t%lu\n", *(unsigned long*) sum_dot_gpu); 
break;
case meta_in:
printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%d]\n",
(*(int*) sum_dot_gpu
== (int) ((ni - 2) * (nj - 2) * (nk - 2) * iters) ?
"PASSED" : "FAILED"),
(ni - 2) * (nj - 2) * (nk - 2) * iters, (*(int*) sum_dot_gpu)); 
printf("Test Dot-Product:\t%d\n", *(int*) sum_dot_gpu); 
break;
case meta_ui:
printf("Test Dot-Product:\t%s\n\tExpect[%d] Returned[%d]\n",
(*(unsigned int*) sum_dot_gpu
== (unsigned int) ((ni - 2) * (nj - 2) * (nk - 2)
* iters) ? "PASSED" : "FAILED"),
(ni - 2) * (nj - 2) * (nk - 2) * iters,
(*(unsigned int*) sum_dot_gpu)); 
printf("Test Dot-Product:\t%d\n", *(unsigned int*) sum_dot_gpu); 
break;
}
#endif 
#ifdef KERNEL_STENCIL
printf("Stencil \n");
gettimeofday(&start, NULL);
for (iter = 0; iter < iters; iter++)
ret = meta_stencil_3d7p(&dimgrid, &dimblock, dev_data3, dev_data3_2,
&dimarray, &arr_start, &arr_end, g_type, false);
gettimeofday(&end, NULL);
fprintf(stderr, "Kernel Status: %d\n", ret);
printf("stencil_3d7p Kern time: %f\n",
((end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec)) / (iters));
#endif 
#ifdef KERNEL_REDUCE
istat = meta_copy_h2d(reduction, zero, g_typesize, true);
ret = meta_reduce(&dimgrid, &dimblock, dev_data3_2, &dimarray, &arr_start,
&arr_end, reduction, g_type, false);
istat = meta_copy_d2h(sum_dot_gpu, reduction, g_typesize, false);
#endif 
switch (g_type) {
case meta_db:
printf("Test stencil_3d7p:\t%f\n", *(double*) sum_dot_gpu); 
break;
case meta_fl:
printf("Test stencil_3d7p:\t%f\n", *(float*) sum_dot_gpu); 
break;
case meta_ul:
printf("Test stencil_3d7p:\t%lu\n", *(unsigned long*) sum_dot_gpu); 
break;
case meta_in:
printf("Test stencil_3d7p:\t%d\n", *(int*) sum_dot_gpu); 
break;
case meta_ui:
printf("Test Dot-Product:\t%d\n", *(unsigned int*) sum_dot_gpu); 
break;
}
#ifdef KERNEL_TRANSPOSE
gettimeofday(&start, NULL);
for (iter = 0; iter < iters; iter++)
ret = meta_transpose_face(&dimgrid, &dimblock, dev_data3_2,
dev_data3, &trans_2d, &trans_2d, g_type, false);
gettimeofday(&end, NULL);
fprintf(stderr, "transpose Kernel Status: %d\n", ret);
printf("transpose Kern time: %f\n",
((end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec)) / (iters));
gettimeofday(&start, NULL);
for (iter = 0; iter < iters; iter++)
meta_copy_d2h(data3, dev_data3, g_typesize * ni * nj * nk, false);
gettimeofday(&end, NULL);
printf("D2H time: %f\n",
((end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec)) / (iters));
#endif 
deallocate_(); 
#ifdef WITH_TIMERS
metaTimersFinish();
#endif
return 0;
} 
