#include "crs_prodAAtx.h"
#include "crs_prodAAtx_types.h"
#include "m2c.h"
#include "omp.h"
#include <math.h>
static void b_crs_prodAAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b);
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
int ncols, int nrhs, boolean_T varargin_1);
static void b_crs_prodAx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int nrhs, boolean_T varargin_1);
static void b_m2c_error(void);
static void c_crs_prodAAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b);
static void c_m2c_error(void);
static void crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows, int A_ncols,
const emxArray_real_T *x, emxArray_real_T *b);
static void crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int ncols, int nrhs);
static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows, int nrhs);
static void d_m2c_error(void);
static void m2c_error(void);
static void m2c_warn(void);
static void b_crs_prodAAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b)
{
emxArray_real_T *Atx;
int i;
int loop_ub;
emxInit_real_T(&Atx, 2);
i = Atx->size[0] * Atx->size[1];
Atx->size[0] = A_ncols;
Atx->size[1] = x->size[1];
emxEnsureCapacity_real_T(Atx, i);
loop_ub = A_ncols * x->size[1];
for (i = 0; i < loop_ub; i++) {
Atx->data[i] = 0.0;
}
b_crs_prodAtx(A_row_ptr, A_col_ind, A_val, A_nrows, A_ncols, x, Atx);
if (x->size[1] < Atx->size[1]) {
d_m2c_error();
}
i = b->size[0] * b->size[1];
b->size[0] = A_nrows;
b->size[1] = x->size[1];
emxEnsureCapacity_real_T(b, i);
loop_ub = A_nrows * x->size[1];
for (i = 0; i < loop_ub; i++) {
b->data[i] = 0.0;
}
crs_prodAx_kernel(A_row_ptr, A_col_ind, A_val, Atx, Atx->size[0], b, A_nrows,
A_nrows, Atx->size[1]);
emxFree_real_T(&Atx);
}
static void b_crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b)
{
if ((b->size[0] < A_ncols) || (b->size[1] < x->size[1])) {
c_m2c_error();
}
crs_prodAtx_kernel(A_row_ptr, A_col_ind, A_val, x, x->size[0], b, b->size[0],
A_nrows, A_ncols, x->size[1]);
}
static void b_crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
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
static void b_crs_prodAx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows,
int nrhs, boolean_T varargin_1)
{
double t;
int b_i;
int b_remainder;
int chunk;
int i;
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
t = 0.0;
chunk = row_ptr->data[i - 1];
b_i = row_ptr->data[i] - 1;
for (j = chunk; j <= b_i; j++) {
t += val->data[j - 1] * x->data[b_remainder + col_ind->data[j - 1]];
}
b->data[threadID + i] = t;
}
b_remainder += x_m;
threadID += b_m;
}
}
static void b_m2c_error(void)
{
M2C_error("prodAAtx:IncorrectBuffer", "Buffer Atx has incorrect size.");
}
static void c_crs_prodAAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows,
int A_ncols, const emxArray_real_T *x,
emxArray_real_T *b)
{
emxArray_real_T *Atx;
int i;
int loop_ub;
if ((b->size[0] < A_nrows) || (b->size[1] != x->size[1])) {
#pragma omp master
{
m2c_error();
}
}
emxInit_real_T(&Atx, 2);
i = Atx->size[0] * Atx->size[1];
Atx->size[0] = A_ncols;
Atx->size[1] = x->size[1];
emxEnsureCapacity_real_T(Atx, i);
loop_ub = A_ncols * x->size[1];
for (i = 0; i < loop_ub; i++) {
Atx->data[i] = 0.0;
}
b_crs_prodAtx(A_row_ptr, A_col_ind, A_val, A_nrows, A_ncols, x, Atx);
if ((b->size[0] < A_nrows) || (b->size[1] < Atx->size[1])) {
d_m2c_error();
}
crs_prodAx_kernel(A_row_ptr, A_col_ind, A_val, Atx, Atx->size[0], b,
b->size[0], A_nrows, Atx->size[1]);
emxFree_real_T(&Atx);
}
static void c_m2c_error(void)
{
M2C_error("crs_prodAtx:BufferTooSmal",
"Buffer space for output b is too small.");
}
static void crs_prodAtx(const emxArray_int32_T *A_row_ptr,
const emxArray_int32_T *A_col_ind,
const emxArray_real_T *A_val, int A_nrows, int A_ncols,
const emxArray_real_T *x, emxArray_real_T *b)
{
int n;
if ((b->size[0] < A_ncols) || (b->size[1] < x->size[1])) {
c_m2c_error();
}
n = omp_get_num_threads();
if (n == 1) {
crs_prodAtx_kernel(A_row_ptr, A_col_ind, A_val, x, x->size[0], b,
b->size[0], A_nrows, A_ncols, x->size[1]);
} else {
b_crs_prodAtx_kernel(A_row_ptr, A_col_ind, A_val, x, x->size[0], b,
b->size[0], A_nrows, A_ncols, x->size[1], true);
}
}
static void crs_prodAtx_kernel(const emxArray_int32_T *row_ptr,
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
static void crs_prodAx_kernel(const emxArray_int32_T *row_ptr,
const emxArray_int32_T *col_ind,
const emxArray_real_T *val,
const emxArray_real_T *x, int x_m,
emxArray_real_T *b, int b_m, int nrows, int nrhs)
{
double t;
int b_i;
int boffset;
int i;
int i1;
int j;
int k;
int xoffset;
xoffset = -1;
boffset = 0;
for (k = 0; k < nrhs; k++) {
for (i = 0; i < nrows; i++) {
t = 0.0;
b_i = row_ptr->data[i];
i1 = row_ptr->data[i + 1] - 1;
for (j = b_i; j <= i1; j++) {
t += val->data[j - 1] * x->data[xoffset + col_ind->data[j - 1]];
}
b->data[boffset + i] = t;
}
xoffset += x_m;
boffset += b_m;
}
}
static void d_m2c_error(void)
{
M2C_error("crs_prodAx:BufferTooSmal",
"Buffer space for output b is too small.");
}
static void m2c_error(void)
{
M2C_error("crs_prodAAtx:IncorrectBuffer", "Buffer b has incorrect size.");
}
static void m2c_warn(void)
{
M2C_warn("crs_prodAAtx:NestedParallel",
"You are trying to use nested parallel regions, but nested "
"parallelism is not enabled.");
}
void crs_prodAAtx(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b, emxArray_real_T *Atx,
const emxArray_int32_T *nthreads)
{
int n;
boolean_T ismt;
if ((b->size[0] < A->nrows) || (b->size[1] != x->size[1])) {
#pragma omp master
{
m2c_error();
}
}
if ((Atx->size[0] < A->ncols) || (Atx->size[1] != x->size[1])) {
#pragma omp master
{
b_m2c_error();
}
}
n = omp_get_num_threads();
ismt = (n > 1);
if (nthreads->size[0] != 0) {
n = omp_get_nested();
if ((n == 0) && ismt && (nthreads->data[0] > 1)) {
#pragma omp master
{
m2c_warn();
}
}
#pragma omp parallel default(shared) num_threads(nthreads->data[0])
{
crs_prodAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, Atx);
#pragma omp barrier
if ((b->size[0] < A->nrows) || (b->size[1] < Atx->size[1])) {
d_m2c_error();
}
n = omp_get_num_threads();
if (n == 1) {
crs_prodAx_kernel(A->row_ptr, A->col_ind, A->val, Atx, Atx->size[0], b,
b->size[0], A->nrows, Atx->size[1]);
} else {
b_crs_prodAx_kernel(A->row_ptr, A->col_ind, A->val, Atx, Atx->size[0], b,
b->size[0], A->nrows, Atx->size[1], true);
}
}
} else if (ismt) {
crs_prodAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, Atx);
#pragma omp barrier
if ((b->size[0] < A->nrows) || (b->size[1] < Atx->size[1])) {
d_m2c_error();
}
n = omp_get_num_threads();
if (n == 1) {
crs_prodAx_kernel(A->row_ptr, A->col_ind, A->val, Atx, Atx->size[0], b,
b->size[0], A->nrows, Atx->size[1]);
} else {
b_crs_prodAx_kernel(A->row_ptr, A->col_ind, A->val, Atx, Atx->size[0], b,
b->size[0], A->nrows, Atx->size[1], true);
}
} else {
b_crs_prodAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, Atx);
if ((b->size[0] < A->nrows) || (b->size[1] < Atx->size[1])) {
d_m2c_error();
}
crs_prodAx_kernel(A->row_ptr, A->col_ind, A->val, Atx, Atx->size[0], b,
b->size[0], A->nrows, Atx->size[1]);
}
}
void crs_prodAAtx_initialize(void)
{
}
void crs_prodAAtx_ser(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b)
{
b_crs_prodAAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, b);
}
void crs_prodAAtx_ser1(const CRS_Matrix *A, const emxArray_real_T *x,
emxArray_real_T *b)
{
c_crs_prodAAtx(A->row_ptr, A->col_ind, A->val, A->nrows, A->ncols, x, b);
}
void crs_prodAAtx_terminate(void)
{
}
