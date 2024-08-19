#include "crs_prodAtx.h"
#include "crs_prodAtx_types.h"
#include "m2c.h"
#include "omp.h"
#include <math.h>
static void b_crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b);
static void b_crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int ncols, int nrhs);
static void c_crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b);
static void crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int ncols, int nrhs, boolean_T varargin_1);
static void m2c_error(void);
static void m2c_warn(void);
static void b_crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b)
{
int i;
i = b->size[0] * b->size[1];
b->size[0] = A_ncols;
b->size[1] = x->size[1];
emxEnsureCapacity_real_T(b, i);
b_crs_prodAtx_kernel(A_row_ptr, A_col_ind, A_val, x, x->size[0], b, A_ncols,
A_nrows, A_ncols, x->size[1]);
}
static void b_crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int ncols, int nrhs)
{
double alpha;
int b_i;
int boffset;
int i;
int i1;
int j;
int k;
int r;
int xoffset;
boffset = 0;
if (1 <= nrows) {
xoffset = 0;
for (k = 0; k < nrhs; k++) {
i = boffset + 1;
i1 = boffset + ncols;
for (j = i; j <= i1; j++) {
b->data[j - 1] = 0.0;
}
for (b_i = 0; b_i < nrows; b_i++) {
alpha = x->data[b_i + xoffset];
i = row_ptr->data[b_i];
i1 = row_ptr->data[b_i + 1] - 1;
for (j = i; j <= i1; j++) {
r = (boffset + col_ind->data[j - 1]) - 1;
b->data[r] += alpha * val->data[j - 1];
}
}
xoffset += x_m;
boffset += b_m;
}
}
}
static void c_crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b)
{
if ((b->size[0] < A_ncols) || (b->size[1] < x->size[1])) {
m2c_error();
}
b_crs_prodAtx_kernel(A_row_ptr, A_col_ind, A_val, x, x->size[0], b,
b->size[0], A_nrows, A_ncols, x->size[1]);
}
static void crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int ncols, int nrhs, boolean_T varargin_1)
{
double alpha;
int b_i;
int b_remainder;
int boffset;
int chunk;
int i;
int iend;
int istart;
int j;
int k;
int nthreads;
int xoffset;
if (varargin_1) {
xoffset = omp_get_num_threads();
nthreads = (int)floor((double)b_m / (double)ncols);
if (xoffset < nthreads) {
nthreads = xoffset;
}
xoffset = omp_get_thread_num();
boffset = xoffset * ncols;
if (nthreads == 1) {
istart = 0;
iend = nrows;
} else {
iend = omp_get_thread_num();
if (iend >= nthreads) {
istart = 0;
iend = 0;
} else {
chunk = nrows / nthreads;
b_remainder = nrows - nthreads * chunk;
if (b_remainder < iend) {
xoffset = b_remainder;
} else {
xoffset = iend;
}
istart = iend * chunk + xoffset;
iend = (istart + chunk) + (iend < b_remainder);
}
}
istart++;
} else {
nthreads = 1;
boffset = 0;
istart = 1;
iend = nrows;
}
if (istart <= iend) {
xoffset = -1;
for (k = 0; k < nrhs; k++) {
chunk = boffset + 1;
i = boffset + ncols;
for (j = chunk; j <= i; j++) {
b->data[j - 1] = 0.0;
}
for (b_i = istart; b_i <= iend; b_i++) {
alpha = x->data[b_i + xoffset];
chunk = row_ptr->data[b_i - 1];
i = row_ptr->data[b_i] - 1;
for (j = chunk; j <= i; j++) {
b_remainder = (boffset + col_ind->data[j - 1]) - 1;
b->data[b_remainder] += alpha * val->data[j - 1];
}
}
xoffset += x_m;
boffset += b_m;
}
}
if (nthreads > 1) {
#pragma omp barrier
xoffset = omp_get_num_threads();
if (xoffset == 1) {
istart = 0;
iend = ncols;
} else {
iend = omp_get_thread_num();
chunk = ncols / xoffset;
b_remainder = ncols - xoffset * chunk;
if (b_remainder < iend) {
xoffset = b_remainder;
} else {
xoffset = iend;
}
istart = iend * chunk + xoffset;
iend = (istart + chunk) + (iend < b_remainder);
}
xoffset = ncols;
for (j = 2; j <= nthreads; j++) {
boffset = 1;
for (k = 0; k < nrhs; k++) {
chunk = boffset + istart;
i = (boffset + iend) - 1;
for (b_i = chunk; b_i <= i; b_i++) {
b->data[b_i - 1] += b->data[(xoffset + b_i) - 1];
}
boffset += b_m;
}
xoffset += ncols;
}
}
}
static void m2c_error(void)
{
M2C_error("crs_prodAtx:BufferTooSmal",
"Buffer space for output b is too small.");
}
static void m2c_warn(void)
{
M2C_warn("crs_prodAtx:NestedParallel",
"You are trying to use nested parallel regions. Solution may be "
"incorrect.");
}
void crs_prodAtx(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b, const emxArray_int32_T *nthreads)
{
int n;
if ((b->size[0] < A->ncols) || (b->size[1] < x->size[1])) {
m2c_error();
}
if (nthreads->size[0] != 0) {
n = omp_get_nested();
if (n == 0) {
n = omp_get_num_threads();
if ((n > 1) && (nthreads->data[0] > 1)) {
#pragma omp master
{
m2c_warn();
}
}
}
#pragma omp parallel default(shared) num_threads(nthreads->data[0])
{
n = omp_get_num_threads();
crs_prodAtx_kernel(A->row_ptr, A->col_ind, A->val, x, x->size[0], b,
b->size[0], A->nrows, A->ncols, x->size[1], n > 1);
}
} else {
n = omp_get_num_threads();
if (n == 1) {
b_crs_prodAtx_kernel(A->row_ptr, A->col_ind, A->val, x, x->size[0], b,
b->size[0], A->nrows, A->ncols, x->size[1]);
} else {
crs_prodAtx_kernel(A->row_ptr, A->col_ind, A->val, x, x->size[0], b,
b->size[0], A->nrows, A->ncols, x->size[1], true);
}
}
}
void crs_prodAtx_initialize(void)
{
}
void crs_prodAtx_ser(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b)
{
b_crs_prodAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, b);
}
void crs_prodAtx_ser1(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b)
{
c_crs_prodAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, b);
}
void crs_prodAtx_terminate(void)
{
}
