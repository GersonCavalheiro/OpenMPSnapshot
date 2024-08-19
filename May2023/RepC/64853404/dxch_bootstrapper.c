#include <stdio.h>
#include <stdlib.h>
#include "metamorph.h"
#define SUM_FACE(a,b,c) ((((a)*(b))*((c)+(((a)+(b))/2.0)-1)))
meta_type_id g_type;
size_t g_typesize;
void set_type(meta_type_id type) {
switch (type) {
case meta_db:
g_typesize = sizeof(double);
break;
case meta_fl:
g_typesize = sizeof(float);
break;
case meta_ul:
g_typesize = sizeof(unsigned long);
break;
case meta_in:
g_typesize = sizeof(int);
break;
case meta_ui:
g_typesize = sizeof(unsigned int);
break;
default:
fprintf(stderr, "Unsupported type: [%d] provided to set_type!\n", type);
exit(-1);
break;
}
g_type = type;
}
void *data3, *face[6];
void *reduction, *dev_data3, *dev_face[6];
void data_allocate(int i, int j, int k) {
a_err istat = 0;
istat = meta_alloc(&reduction, g_typesize);
printf("Status (Sum alloc):\t%d\n", istat);
data3 = malloc(g_typesize * i * j * k);
istat = meta_alloc(&dev_data3, g_typesize * i * j * k);
printf("Status (3D alloc):\t%d\n", istat);
face[0] = malloc(g_typesize * i * j);
face[1] = malloc(g_typesize * i * j);
istat = meta_alloc(&dev_face[0], g_typesize * i * j);
istat |= meta_alloc(&dev_face[1], g_typesize * i * j);
printf("Status (N/S faces):\t%d\n", istat);
face[2] = malloc(g_typesize * j * k);
face[3] = malloc(g_typesize * j * k);
istat = meta_alloc(&dev_face[2], g_typesize * j * k);
istat |= meta_alloc(&dev_face[3], g_typesize * j * k);
printf("Status (E/W faces):\t%d\n", istat);
face[4] = malloc(g_typesize * i * k);
face[5] = malloc(g_typesize * i * k);
istat = meta_alloc(&dev_face[4], g_typesize * i * k);
istat |= meta_alloc(&dev_face[5], g_typesize * i * k);
printf("Status (T/B faces):\t%d\n", istat);
printf("Data Allocated\n");
}
void data_initialize(int ni, int nj, int nk) {
int i, j, k;
switch (g_type) {
default:
case meta_db: {
double *l_data3 = (double *) data3;
for (i = ni - 1; i >= 0; i--) {
for (j = nj - 1; j >= 0; j--) {
for (k = nk - 1; k >= 0; k--) {
if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
|| k == nk - 1) {
l_data3[i + j * ni + k * ni * nj] = i + j + k;
} else {
l_data3[i + j * ni + k * ni * nj] = 0.0f;
}
}
}
}
}
break;
case meta_fl: {
float *l_data3 = (float *) data3;
for (i = ni - 1; i >= 0; i--) {
for (j = nj - 1; j >= 0; j--) {
for (k = nk - 1; k >= 0; k--) {
if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
|| k == nk - 1) {
l_data3[i + j * ni + k * ni * nj] = i + j + k;
} else {
l_data3[i + j * ni + k * ni * nj] = 0.0f;
}
}
}
}
}
break;
case meta_ul: {
unsigned long *l_data3 = (unsigned long *) data3;
for (i = ni - 1; i >= 0; i--) {
for (j = nj - 1; j >= 0; j--) {
for (k = nk - 1; k >= 0; k--) {
if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
|| k == nk - 1) {
l_data3[i + j * ni + k * ni * nj] = i + j + k;
} else {
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
}
break;
case meta_in: {
int *l_data3 = (int *) data3;
for (i = ni - 1; i >= 0; i--) {
for (j = nj - 1; j >= 0; j--) {
for (k = nk - 1; k >= 0; k--) {
if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
|| k == nk - 1) {
l_data3[i + j * ni + k * ni * nj] = i + j + k;
} else {
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
}
break;
case meta_ui: {
unsigned int *l_data3 = (unsigned int *) data3;
for (i = ni - 1; i >= 0; i--) {
for (j = nj - 1; j >= 0; j--) {
for (k = nk - 1; k >= 0; k--) {
if (i == 0 || i == ni - 1 || j == 0 || j == nj - 1 || k == 0
|| k == nk - 1) {
l_data3[i + j * ni + k * ni * nj] = i + j + k;
} else {
l_data3[i + j * ni + k * ni * nj] = 0;
}
}
}
}
}
break;
}
}
void gpu_initialize(void) {
int istat, deviceused; 
#ifdef WITH_CUDA
int idevice = 0;
istat = meta_set_acc(idevice, metaModePreferCUDA); 
#endif
#ifdef WITH_OPENCL
int idevice = -1;
istat = meta_set_acc(idevice, metaModePreferGeneric); 
#endif
#ifdef WITH_OPENMP
int idevice = 0;
istat = meta_set_acc(idevice, metaModePreferOpenMP); 
#endif
meta_preferred_mode mode;
istat = meta_get_acc(&deviceused, &mode); 
} 
int check_face_sum(void * sum, int a, int b, int c) {
printf("CHECK: %d %d %d\n", a, b, c);
int ret = 0;
switch (g_type) {
case meta_db:
if (SUM_FACE(a,b,c) != *(double *) sum) {
fprintf(stderr,
"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%f]\n",
(double) SUM_FACE(a, b, c), *(double *) sum);
ret = -1;
}
break;
case meta_fl:
if (SUM_FACE(a,b,c) != *(float *) sum) {
fprintf(stderr,
"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%f]\n",
(float) SUM_FACE(a, b, c), *(float *) sum);
ret = -1;
}
break;
case meta_ul:
if (SUM_FACE(a,b,c) != (float) (*(unsigned long *) sum)) {
fprintf(stderr,
"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%lu]\n",
SUM_FACE(a, b, c), *(unsigned long *) sum);
ret = -1;
}
break;
case meta_in:
if (SUM_FACE(a,b,c) != (float) (*(int *) sum)) {
fprintf(stderr,
"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%d]\n",
SUM_FACE(a, b, c), *(int *) sum);
ret = -1;
}
break;
case meta_ui:
if (SUM_FACE(a,b,c) != (float) (*(unsigned int *) sum)) {
fprintf(stderr,
"Error: sum doesn't match!\n\tExpected: [%f]\t Returned: [%d]\n",
SUM_FACE(a, b, c), *(unsigned int *) sum);
ret = -1;
}
break;
default:
ret = -1;
break;
}
return ret;
}
meta_face * make_slab2d_from_3d(int face, int ni, int nj, int nk,
int thickness) {
meta_face * ret = (meta_face*) malloc(
sizeof(meta_face));
ret->count = 3;
ret->size = (int*) malloc(sizeof(int) * 3);
ret->stride = (int*) malloc(sizeof(int) * 3);
if (face & 1) {
if (face == 1)
ret->start = ni * nj * (nk - thickness);
if (face == 3)
ret->start = ni - thickness;
if (face == 5)
ret->start = ni * (nj - thickness);
} else
ret->start = 0;
ret->size[0] = nk, ret->size[1] = nj, ret->size[2] = ni;
if (face < 2)
ret->size[0] = thickness;
if (face > 3)
ret->size[1] = thickness;
if (face > 1 && face < 4)
ret->size[2] = thickness;
ret->stride[0] = ni * nj, ret->stride[1] = ni, ret->stride[2] = 1;
printf(
"Generated Face:\n\tcount: %d\n\tstart: %d\n\tsize: %d %d %d\n\tstride: %d %d %d\n",
ret->count, ret->start, ret->size[0], ret->size[1], ret->size[2],
ret->stride[0], ret->stride[1], ret->stride[2]);
return ret;
}
meta_face * make_face(int face, int ni, int nj, int nk) {
return make_slab2d_from_3d(face, ni, nj, nk, 1);
}
void check_dims(meta_dim3 dim, meta_dim3 s, meta_dim3 e) {
printf(
"Integrity check dim(%ld, %ld, %ld) start(%ld, %ld, %ld) end(%ld, %ld, %ld)\n",
dim[0], dim[1], dim[2], s[0], s[1], s[2], e[0], e[1], e[2]);
}
void check_buffer(void* h_buf, void * d_buf, int leng) {
meta_copy_d2h(h_buf, d_buf, g_typesize * leng, 0);
int i;
double sum = 0.0;
for (i = 0; i < leng; i++) {
printf("%f\n", ((double*) h_buf)[i]);
sum += ((double*) h_buf)[i];
}
printf("SUM: %f\n", sum);
}
int check_fp(double expect, double test, double tol) {
return (abs((expect - test) / expect) < tol);
}
int main(int argc, char **argv) {
#ifdef __DEBUG__
int breakMe = 0;
while (breakMe);
#endif
int i = argc;
int ni, nj, nk, tx, ty, tz, face_id, l_type;
a_bool async, autoconfig;
meta_face * face_spec;
meta_dim3 dimgrid_red, dimblock_red, dimgrid_tr_red, dimarray_3d, arr_start,
arr_end, dim_array2d, start_2d, end_2d, trans_dim, rtrans_dim;
if (i < 11) {
printf(
"<ni> <nj> <nk> <tblockx> <tblocky> <tblockz> <face> <data_type> <async> <autoconfig>\n");
return (1);
}
ni = atoi(argv[1]);
nj = atoi(argv[2]);
nk = atoi(argv[3]);
tx = atoi(argv[4]);
ty = atoi(argv[5]);
tz = atoi(argv[6]);
face_id = atoi(argv[7]);
l_type = atoi(argv[8]);
set_type((meta_type_id) l_type);
async = (a_bool) atoi(argv[9]);
autoconfig = (a_bool) atoi(argv[10]);
#ifdef WITH_OPENMP
#pragma omp parallel
{
#pragma omp master
printf("num threads %d\n", omp_get_num_threads());
}
#endif
dimblock_red[0] = tx, dimblock_red[1] = ty, dimblock_red[2] = tz;
dimgrid_red[0] = ni / tx + ((ni % tx) ? 1 : 0);
dimgrid_red[1] = nj / ty + ((nj % ty) ? 1 : 0);
dimgrid_red[2] = nk / tz + ((nk % tz) ? 1 : 0);
dimarray_3d[0] = ni, dimarray_3d[1] = nj, dimarray_3d[2] = nk;
void * sum_gpu, *zero;
sum_gpu = malloc(g_typesize);
zero = malloc(g_typesize);
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
#ifdef WITH_TIMERS
metaTimersInit();
#endif
gpu_initialize();
data_allocate(ni, nj, nk);
data_initialize(ni, nj, nk);
struct timeval start, end;
for (i = 0; i < 1; i++) {
meta_copy_h2d(dev_data3, data3, ni * nj * nk * g_typesize, async);
for (;
meta_validate_worksize(&dimgrid_red, &dimblock_red) != 0
&& dimblock_red[2] > 1;
dimgrid_red[2] <<= 1, dimblock_red[2] >>= 1)
;
meta_copy_h2d(reduction, zero, g_typesize, async);
arr_start[0] = ((face_id == 3) ? ni - 1 : 0);
arr_end[0] = ((face_id == 2) ? 0 : ni - 1);
arr_start[1] = ((face_id == 5) ? nj - 1 : 0);
arr_end[1] = ((face_id == 4) ? 0 : nj - 1);
arr_start[2] = ((face_id == 1) ? nk - 1 : 0);
arr_end[2] = ((face_id == 0) ? 0 : nk - 1);
a_err ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
autoconfig ? NULL : &dimblock_red, dev_data3, &dimarray_3d,
&arr_start, &arr_end, reduction, g_type, async);
printf("Reduce Error: %d\n", ret);
meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
printf("Initial Face Integrity Check: %s\n",
check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
(face_id < 2 || face_id > 3 ? ni : nk),
(face_id & 1 ?
(face_id < 2 ?
nk - 1 : (face_id < 4 ? ni - 1 : nj - 1)) :
0)) ? "FAILED" : "PASSED");
face_spec = make_face(face_id, ni, nj, nk);
gettimeofday(&start, NULL);
ret = meta_pack_face(NULL, NULL, dev_face[face_id], dev_data3,
face_spec, g_type, async);
gettimeofday(&end, NULL);
printf("Pack Return Val: %d\n", ret);
printf("pack time: %f\n",
(end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec));
meta_copy_h2d(reduction, zero, g_typesize, async);
dim_array2d[0] = face_spec->size[2], dim_array2d[1] =
face_spec->size[1], dim_array2d[2] = face_spec->size[0];
start_2d[0] = start_2d[1] = start_2d[2] = 0;
end_2d[0] = (dim_array2d[0] == 1 ? 0 : ni - 1);
end_2d[1] = (dim_array2d[1] == 1 ? 0 : nj - 1);
end_2d[2] = (dim_array2d[2] == 1 ? 0 : nk - 1);
ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
autoconfig ? NULL : &dimblock_red, dev_face[face_id],
&dim_array2d, &start_2d, &end_2d, reduction, g_type, async);
meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
printf("Packed Face Integrity Check: %s\n",
check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
(face_id < 2 || face_id > 3 ? ni : nk),
(face_id & 1 ?
(face_id < 2 ?
nk - 1 : (face_id < 4 ? ni - 1 : nj - 1)) :
0)) ? "FAILED" : "PASSED");
gettimeofday(&start, NULL);
ret = meta_unpack_face(NULL, NULL, dev_face[face_id], dev_data3,
face_spec, g_type, async);
gettimeofday(&end, NULL);
printf("unpack Return Val: %d\n", ret);
printf("unpack time: %f\n",
(end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec));
meta_copy_h2d(reduction, zero, g_typesize, async);
arr_start[0] = ((face_id == 3) ? ni - 1 : 0);
arr_end[0] = ((face_id == 2) ? 0 : ni - 1);
arr_start[1] = ((face_id == 5) ? nj - 1 : 0);
arr_end[1] = ((face_id == 4) ? 0 : nj - 1);
arr_start[2] = ((face_id == 1) ? nk - 1 : 0);
arr_end[2] = ((face_id == 0) ? 0 : nk - 1);
ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
autoconfig ? NULL : &dimblock_red, dev_data3, &dimarray_3d,
&arr_start, &arr_end, reduction, g_type, async);
meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
printf("RecvAndUnpacked Face Integrity Check: %s\n",
check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
(face_id < 2 || face_id > 3 ? ni : nk),
(face_id & 1 ?
(face_id < 2 ?
nk - 1 : (face_id < 4 ? ni - 1 : nj - 1)) :
0)) ? "FAILED" : "PASSED");
trans_dim[0] = (
face_spec->size[2] == 1 ?
face_spec->size[1] : face_spec->size[2]);
trans_dim[1] = (
face_spec->size[0] == 1 ?
face_spec->size[1] : face_spec->size[0]);
trans_dim[2] = 1;
rtrans_dim[0] = trans_dim[1];
rtrans_dim[1] = trans_dim[0];
rtrans_dim[2] = 1;
void * stuff = calloc(
face_spec->size[0] * face_spec->size[1] * face_spec->size[2],
g_typesize);
meta_copy_h2d(dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
stuff,
g_typesize * face_spec->size[0] * face_spec->size[1]
* face_spec->size[2], async);
gettimeofday(&start, NULL);
ret = meta_transpose_face(NULL, NULL, dev_face[face_id],
dev_face[(face_id & 1) ? face_id - 1 : face_id + 1], &trans_dim,
&trans_dim, g_type, async);
gettimeofday(&end, NULL);
printf("transpose time: %f\n",
(end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec));
printf("Transpose error: %d\n", ret);
meta_copy_h2d(reduction, zero, g_typesize, async);
rtrans_dim[0] = trans_dim[1];
rtrans_dim[1] = trans_dim[0];
rtrans_dim[2] = 1;
start_2d[0] = start_2d[1] = start_2d[2] = 0;
end_2d[0] = trans_dim[0] - 1, end_2d[1] = trans_dim[1] - 1, end_2d[2] =
0;
ret = meta_reduce(autoconfig ? NULL : &dimgrid_red,
autoconfig ? NULL : &dimblock_red,
dev_face[(face_id & 1) ? face_id - 1 : face_id + 1], &trans_dim,
&start_2d, &end_2d, reduction, g_type, async);
meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
printf("Transposed Face Integrity Check: %s\n",
check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
(face_id < 2 || face_id > 3 ? ni : nk),
(face_id & 1 ?
(face_id < 2 ?
nk - 1 : (face_id < 4 ? ni - 1 : nj - 1)) :
0)) ? "FAILED" : "PASSED");
gettimeofday(&start, NULL);
ret = meta_transpose_face(NULL, NULL,
dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
dev_face[face_id], &rtrans_dim, &rtrans_dim, g_type, async);
gettimeofday(&end, NULL);
printf("transpose time: %f\n",
(end.tv_sec - start.tv_sec) * 1000000.0
+ (end.tv_usec - start.tv_usec));
meta_copy_h2d(reduction, zero, g_typesize, async);
start_2d[0] = start_2d[1] = start_2d[2] = 0;
end_2d[0] = rtrans_dim[0] - 1, end_2d[1] = rtrans_dim[1] - 1, end_2d[2] =
0;
dimgrid_tr_red[0] = dimgrid_red[1];
dimgrid_tr_red[1] = dimgrid_red[0];
dimgrid_tr_red[2] = dimgrid_red[2];
ret = meta_reduce(autoconfig ? NULL : &dimgrid_tr_red,
autoconfig ? NULL : &dimblock_red,
dev_face[(face_id & 1) ? face_id - 1 : face_id + 1],
&rtrans_dim, &start_2d, &end_2d, reduction, g_type, async);
meta_copy_d2h(sum_gpu, reduction, g_typesize, async);
printf("Retransposed Face Integrity Check: %s\n",
check_face_sum(sum_gpu, (face_id < 4 ? nj : nk),
(face_id < 2 || face_id > 3 ? ni : nk),
(face_id & 1 ?
(face_id < 2 ?
nk - 1 : (face_id < 4 ? ni - 1 : nj - 1)) :
0)) ? "FAILED" : "PASSED");
;
}
#ifdef WITH_TIMERS
metaTimersFinish();
#endif 
}
