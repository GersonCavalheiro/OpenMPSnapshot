#include "crs_Axpy.h"
#include "crs_Axpy_types.h"
#include "m2c.h"
#include "omp.h"
static void b_crs_Axpy_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *y, int b_m, int nrows, int nrhs);
static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *y, int b_m, int nrows, int nrhs,
boolean_T varargin_1);
static void m2c_error(void);
static void m2c_warn(void);
static void b_crs_Axpy_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *y, int b_m, int nrows, int nrhs)
{
double t;
int b_i;
int boffset;
int i;
int i1;
int j;
int k;
int t_tmp;
int xoffset;
xoffset = -1;
boffset = -1;
for (k = 0; k < nrhs; k++) {
for (i = 0; i < nrows; i++) {
t_tmp = (boffset + i) + 1;
t = y->data[t_tmp];
b_i = row_ptr->data[i];
i1 = row_ptr->data[i + 1] - 1;
for (j = b_i; j <= i1; j++) {
t += val->data[j - 1] * x->data[xoffset + col_ind->data[j - 1]];
}
y->data[t_tmp] = t;
}
xoffset += x_m;
boffset += b_m;
}
}
static void crs_Axpy_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *y, int b_m, int nrows, int nrhs,
boolean_T varargin_1)
{
double t;
int b_i;
int b_remainder;
int chunk;
int i;
int i1;
int iend;
int istart;
int j;
int k;
int threadID;
if (varargin_1) {
iend = omp_get_num_threads();
if (iend == 1) {
istart = 0;
iend = nrows;
} else {
threadID = omp_get_thread_num();
chunk = nrows / iend;
b_remainder = nrows - iend * chunk;
if (b_remainder < threadID) {
iend = b_remainder;
} else {
iend = threadID;
}
istart = threadID * chunk + iend;
iend = (istart + chunk) + (threadID < b_remainder);
}
} else {
istart = 0;
iend = nrows;
}
b_remainder = -1;
threadID = -1;
for (k = 0; k < nrhs; k++) {
for (i = istart + 1; i <= iend; i++) {
chunk = threadID + i;
t = y->data[chunk];
b_i = row_ptr->data[i - 1];
i1 = row_ptr->data[i] - 1;
for (j = b_i; j <= i1; j++) {
t += val->data[j - 1] * x->data[b_remainder + col_ind->data[j - 1]];
}
y->data[chunk] = t;
}
b_remainder += x_m;
threadID += b_m;
}
}
static void m2c_error(void)
{
M2C_error("crs_Axpy:BufferTooSmal",
"Buffer space for output y is too small.");
}
static void m2c_warn(void)
{
M2C_warn("crs_Axpy:NestedParallel", "You are trying to use nested parallel "
"regions. Solution may be incorrect.");
}
void crs_Axpy(const CRS_Matrix *A, const emxArray_real_T *x, emxArray_real_T *y,
int nthreads)
{
int n;
if ((y->size[0] < A->nrows) || (y->size[1] < x->size[1])) {
m2c_error();
}
n = omp_get_nested();
if (n == 0) {
n = omp_get_num_threads();
if ((n > 1) && (nthreads > 1)) {
#pragma omp master
{
m2c_warn();
}
}
}
#pragma omp parallel default(shared) num_threads(nthreads)
{
n = omp_get_num_threads();
crs_Axpy_kernel(A->row_ptr, A->col_ind, A->val, x, x->size[0], y, y->size[0],
A->nrows, x->size[1], n > 1);
}
}
void crs_Axpy_initialize(void)
{
}
void crs_Axpy_ser1(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b)
{
if ((b->size[0] < A->nrows) || (b->size[1] < x->size[1])) {
m2c_error();
}
b_crs_Axpy_kernel(A->row_ptr, A->col_ind, A->val, x, x->size[0], b,
b->size[0], A->nrows, x->size[1]);
}
void crs_Axpy_terminate(void)
{
}
