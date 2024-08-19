#include "base.h"
#include "matrix.h"
#include "util.h"
#include "timer.h"
#include "splatt_lapack.h"
#include <math.h>
static void p_form_gram(
matrix_t * neq_matrix,
matrix_t * * aTa,
idx_t const mode,
idx_t const nmodes,
val_t const reg)
{
splatt_blas_int N = aTa[0]->J;
val_t * const restrict neqs = neq_matrix->vals;
#pragma omp parallel
{
#pragma omp for schedule(static, 1)
for(splatt_blas_int i=0; i < N; ++i) {
neqs[i+(i*N)] = 1. + reg;
for(splatt_blas_int j=0; j < N; ++j) {
neqs[j+(i*N)] = 1.;
}
}
for(idx_t m=0; m < nmodes; ++m) {
if(m == mode) {
continue;
}
val_t const * const restrict mat = aTa[m]->vals;
#pragma omp for schedule(static, 1)
for(splatt_blas_int i=0; i < N; ++i) {
for(splatt_blas_int j=i; j < N; ++j) {
neqs[j+(i*N)] *= mat[j+(i*N)];
}
}
} 
#pragma omp barrier
#pragma omp for schedule(static, 1)
for(splatt_blas_int i=0; i < N; ++i) {
for(splatt_blas_int j=0; j < i; ++j) {
neqs[j+(i*N)] = neqs[i+(j*N)];
}
}
} 
}
static void p_mat_2norm(
matrix_t * const A,
val_t * const restrict lambda,
rank_info * const rinfo,
thd_info * const thds)
{
idx_t const I = A->I;
idx_t const J = A->J;
val_t * const restrict vals = A->vals;
#pragma omp parallel
{
int const tid = splatt_omp_get_thread_num();
val_t * const mylambda = (val_t *) thds[tid].scratch[0];
for(idx_t j=0; j < J; ++j) {
mylambda[j] = 0;
}
#pragma omp for schedule(static)
for(idx_t i=0; i < I; ++i) {
for(idx_t j=0; j < J; ++j) {
mylambda[j] += vals[j + (i*J)] * vals[j + (i*J)];
}
}
thd_reduce(thds, 0, J, REDUCE_SUM);
#pragma omp master
{
#ifdef SPLATT_USE_MPI
timer_start(&timers[TIMER_MPI_NORM]);
timer_start(&timers[TIMER_MPI_COMM]);
MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);
timer_stop(&timers[TIMER_MPI_COMM]);
timer_stop(&timers[TIMER_MPI_NORM]);
#else
memcpy(lambda, mylambda, J * sizeof(val_t));
#endif
}
#pragma omp barrier
#pragma omp for schedule(static)
for(idx_t j=0; j < J; ++j) {
lambda[j] = sqrt(lambda[j]);
}
#pragma omp for schedule(static)
for(idx_t i=0; i < I; ++i) {
for(idx_t j=0; j < J; ++j) {
vals[j+(i*J)] /= lambda[j];
}
}
} 
}
static void p_mat_maxnorm(
matrix_t * const A,
val_t * const restrict lambda,
rank_info * const rinfo,
thd_info * const thds)
{
idx_t const I = A->I;
idx_t const J = A->J;
val_t * const restrict vals = A->vals;
#pragma omp parallel
{
int const tid = splatt_omp_get_thread_num();
val_t * const mylambda = (val_t *) thds[tid].scratch[0];
for(idx_t j=0; j < J; ++j) {
mylambda[j] = 0;
}
#pragma omp for schedule(static)
for(idx_t i=0; i < I; ++i) {
for(idx_t j=0; j < J; ++j) {
mylambda[j] = SS_MAX(mylambda[j], vals[j+(i*J)]);
}
}
thd_reduce(thds, 0, J, REDUCE_MAX);
#pragma omp master
{
#ifdef SPLATT_USE_MPI
timer_start(&timers[TIMER_MPI_NORM]);
timer_start(&timers[TIMER_MPI_COMM]);
MPI_Allreduce(mylambda, lambda, J, SPLATT_MPI_VAL, MPI_MAX, rinfo->comm_3d);
timer_stop(&timers[TIMER_MPI_COMM]);
timer_stop(&timers[TIMER_MPI_NORM]);
#else
memcpy(lambda, mylambda, J * sizeof(val_t));
#endif
}
#pragma omp barrier
#pragma omp for schedule(static)
for(idx_t j=0; j < J; ++j) {
lambda[j] = SS_MAX(lambda[j], 1.);
}
#pragma omp for schedule(static)
for(idx_t i=0; i < I; ++i) {
for(idx_t j=0; j < J; ++j) {
vals[j+(i*J)] /= lambda[j];
}
}
} 
}
static void p_mat_forwardsolve(
matrix_t const * const L,
matrix_t * const B)
{
idx_t const N = L->I;
val_t const * const restrict lv = L->vals;
val_t * const restrict bv = B->vals;
for(idx_t j=0; j < N; ++j) {
bv[j] /= lv[0];
}
for(idx_t i=1; i < N; ++i) {
for(idx_t j=0; j < i; ++j) {
for(idx_t f=0; f < N; ++f) {
bv[f+(i*N)] -= lv[j+(i*N)] * bv[f+(j*N)];
}
}
for(idx_t f=0; f < N; ++f) {
bv[f+(i*N)] /= lv[i+(i*N)];
}
}
}
static void p_mat_backwardsolve(
matrix_t const * const U,
matrix_t * const B)
{
idx_t const N = U->I;
val_t const * const restrict rv = U->vals;
val_t * const restrict bv = B->vals;
for(idx_t f=0; f < N; ++f) {
idx_t const i = N-1;
bv[f+(i*N)] /= rv[i+(i*N)];
}
for(idx_t row=2; row <= N; ++row) {
idx_t const i = N - row;
for(idx_t j=i+1; j < N; ++j) {
for(idx_t f=0; f < N; ++f) {
bv[f+(i*N)] -= rv[j+(i*N)] * bv[f+(j*N)];
}
}
for(idx_t f=0; f < N; ++f) {
bv[f+(i*N)] /= rv[i+(i*N)];
}
}
}
void mat_syminv(
matrix_t * const A)
{
assert(A->I == A->J);
idx_t const N = A->I;
matrix_t * L = mat_alloc(N, N);
mat_cholesky(A, L);
memset(A->vals, 0, N*N*sizeof(val_t));
for(idx_t n=0; n < N; ++n) {
A->vals[n+(n*N)] = 1.;
}
p_mat_forwardsolve(L, A);
for(idx_t i=0; i < N; ++i) {
for(idx_t j=i+1; j < N; ++j) {
L->vals[j+(i*N)] = L->vals[i+(j*N)];
L->vals[i+(j*N)] = 0.;
}
}
p_mat_backwardsolve(L, A);
mat_free(L);
}
void mat_cholesky(
matrix_t const * const A,
matrix_t * const L)
{
assert(A->I == A->J);
assert(A->I == L->J);
assert(L->I == L->J);
idx_t const N = A->I;
val_t const * const restrict av = A->vals;
val_t * const restrict lv = L->vals;
memset(lv, 0, N*N*sizeof(val_t));
for (idx_t i = 0; i < N; ++i) {
for (idx_t j = 0; j <= i; ++j) {
val_t inner = 0;
for (idx_t k = 0; k < j; ++k) {
inner += lv[k+(i*N)] * lv[k+(j*N)];
}
if(i == j) {
lv[j+(i*N)] = sqrt(av[i+(i*N)] - inner);
} else {
lv[j+(i*N)] = 1.0 / lv[j+(j*N)] * (av[j+(i*N)] - inner);
}
}
}
}
void mat_aTa_hada(
matrix_t ** mats,
idx_t const start,
idx_t const nmults,
idx_t const nmats,
matrix_t * const buf,
matrix_t * const ret)
{
idx_t const F = mats[0]->J;
assert(ret->I == ret->J);
assert(ret->I == F);
assert(buf->I == F);
assert(buf->J == F);
assert(ret->vals != NULL);
assert(mats[0]->rowmajor);
assert(ret->rowmajor);
val_t       * const restrict rv   = ret->vals;
val_t       * const restrict bufv = buf->vals;
for(idx_t i=0; i < F; ++i) {
for(idx_t j=i; j < F; ++j) {
rv[j+(i*F)] = 1.;
}
}
for(idx_t mode=0; mode < nmults; ++mode) {
idx_t const m = (start+mode) % nmats;
idx_t const I  = mats[m]->I;
val_t const * const Av = mats[m]->vals;
memset(bufv, 0, F * F * sizeof(val_t));
for(idx_t i=0; i < I; ++i) {
for(idx_t mi=0; mi < F; ++mi) {
for(idx_t mj=mi; mj < F; ++mj) {
bufv[mj + (mi*F)] += Av[mi + (i*F)] * Av[mj + (i*F)];
}
}
}
for(idx_t mi=0; mi < F; ++mi) {
for(idx_t mj=mi; mj < F; ++mj) {
rv[mj + (mi*F)] *= bufv[mj + (mi*F)];
}
}
}
for(idx_t i=1; i < F; ++i) {
for(idx_t j=0; j < i; ++j) {
rv[j + (i*F)] = rv[i + (j*F)];
}
}
}
void mat_aTa(
matrix_t const * const A,
matrix_t * const ret,
rank_info * const rinfo,
thd_info * const thds,
idx_t const nthreads)
{
timer_start(&timers[TIMER_ATA]);
assert(ret->I == ret->J);
assert(ret->I == A->J);
assert(ret->vals != NULL);
assert(A->rowmajor);
assert(ret->rowmajor);
idx_t const I = A->I;
idx_t const F = A->J;
val_t const * const restrict Av = A->vals;
char uplo = 'L';
char trans = 'N'; 
splatt_blas_int N = (splatt_blas_int) F;
splatt_blas_int K = (splatt_blas_int) I;
splatt_blas_int lda = N;
splatt_blas_int ldc = N;
val_t alpha = 1.;
val_t beta = 0.;
SPLATT_BLAS(syrk)(&uplo, &trans, &N, &K, &alpha, A->vals, &lda, &beta, ret->vals,
&ldc);
#ifdef SPLATT_USE_MPI
timer_start(&timers[TIMER_MPI_ATA]);
timer_start(&timers[TIMER_MPI_COMM]);
MPI_Allreduce(MPI_IN_PLACE, ret->vals, F * F, SPLATT_MPI_VAL, MPI_SUM,
rinfo->comm_3d);
timer_stop(&timers[TIMER_MPI_COMM]);
timer_stop(&timers[TIMER_MPI_ATA]);
#endif
timer_stop(&timers[TIMER_ATA]);
}
void mat_matmul(
matrix_t const * const A,
matrix_t const * const B,
matrix_t  * const C)
{
timer_start(&timers[TIMER_MATMUL]);
assert(A->J == B->I);
assert(C->I * C->J <= A->I * B->J);
C->I = A->I;
C->J = B->J;
val_t const * const restrict av = A->vals;
val_t const * const restrict bv = B->vals;
val_t       * const restrict cv = C->vals;
idx_t const M  = A->I;
idx_t const N  = B->J;
idx_t const Na = A->J;
idx_t const TILE = 16;
#pragma omp parallel for schedule(static)
for(idx_t i=0; i < M; ++i) {
for(idx_t jt=0; jt < N; jt += TILE) {
for(idx_t kt=0; kt < Na; kt += TILE) {
idx_t const JSTOP = SS_MIN(jt+TILE, N);
for(idx_t j=jt; j < JSTOP; ++j) {
val_t accum = 0;
idx_t const KSTOP = SS_MIN(kt+TILE, Na);
for(idx_t k=kt; k < KSTOP; ++k) {
accum += av[k + (i*Na)] * bv[j + (k*N)];
}
cv[j + (i*N)] += accum;
}
}
}
}
timer_stop(&timers[TIMER_MATMUL]);
}
void mat_normalize(
matrix_t * const A,
val_t * const restrict lambda,
splatt_mat_norm const which,
rank_info * const rinfo,
thd_info * const thds,
idx_t const nthreads)
{
timer_start(&timers[TIMER_MATNORM]);
splatt_omp_set_num_threads(nthreads);
switch(which) {
case MAT_NORM_2:
p_mat_2norm(A, lambda, rinfo, thds);
break;
case MAT_NORM_MAX:
p_mat_maxnorm(A, lambda, rinfo, thds);
break;
default:
fprintf(stderr, "SPLATT: mat_normalize supports 2 and MAX only.\n");
abort();
}
timer_stop(&timers[TIMER_MATNORM]);
}
void mat_solve_normals(
idx_t const mode,
idx_t const nmodes,
matrix_t * * aTa,
matrix_t * rhs,
val_t const reg)
{
timer_start(&timers[TIMER_INV]);
splatt_blas_int N = aTa[0]->J;
p_form_gram(aTa[MAX_NMODES], aTa, mode, nmodes, reg);
splatt_blas_int info;
char uplo = 'L';
splatt_blas_int lda = N;
splatt_blas_int ldb = N;
splatt_blas_int order = N;
splatt_blas_int nrhs = (splatt_blas_int) rhs->I;
val_t * const neqs = aTa[MAX_NMODES]->vals;
bool is_spd = true;
SPLATT_BLAS(potrf)(&uplo, &order, neqs, &lda, &info);
if(info) {
fprintf(stderr, "SPLATT: Gram matrix is not SPD. Trying `GELSS`.\n");
is_spd = false;
}
if(is_spd) {
SPLATT_BLAS(potrs)(&uplo, &order, &nrhs, neqs, &lda, rhs->vals, &ldb, &info);
if(info) {
fprintf(stderr, "SPLATT: DPOTRS returned %d\n", info);
}
} else {
p_form_gram(aTa[MAX_NMODES], aTa, mode, nmodes, reg);
splatt_blas_int effective_rank;
val_t * conditions = splatt_malloc(N * sizeof(*conditions));
splatt_blas_int lwork = -1;
val_t rcond = -1.0f;
val_t work_query;
SPLATT_BLAS(gelss)(&N, &N, &nrhs,
neqs, &lda,
rhs->vals, &ldb,
conditions, &rcond, &effective_rank,
&work_query, &lwork, &info);
lwork = (splatt_blas_int) work_query;
val_t * work = splatt_malloc(lwork * sizeof(*work));
SPLATT_BLAS(gelss)(&N, &N, &nrhs,
neqs, &lda,
rhs->vals, &ldb,
conditions, &rcond, &effective_rank,
work, &lwork, &info);
if(info) {
printf("SPLATT: DGELSS returned %d\n", info);
}
printf("SPLATT:   DGELSS effective rank: %d\n", effective_rank);
splatt_free(conditions);
splatt_free(work);
}
timer_stop(&timers[TIMER_INV]);
}
void calc_gram_inv(
idx_t const mode,
idx_t const nmodes,
matrix_t ** aTa)
{
timer_start(&timers[TIMER_INV]);
idx_t const rank = aTa[0]->J;
val_t * const restrict av = aTa[MAX_NMODES]->vals;
for(idx_t x=0; x < rank*rank; ++x) {
av[x] = 1.;
}
for(idx_t m=1; m < nmodes; ++m) {
idx_t const madjust = (mode + m) % nmodes;
val_t const * const vals = aTa[madjust]->vals;
for(idx_t x=0; x < rank*rank; ++x) {
av[x] *= vals[x];
}
}
mat_syminv(aTa[MAX_NMODES]);
timer_stop(&timers[TIMER_INV]);
}
matrix_t * mat_alloc(
idx_t const nrows,
idx_t const ncols)
{
matrix_t * mat = (matrix_t *) splatt_malloc(sizeof(matrix_t));
mat->I = nrows;
mat->J = ncols;
mat->vals = (val_t *) splatt_malloc(nrows * ncols * sizeof(val_t));
mat->rowmajor = 1;
return mat;
}
matrix_t * mat_rand(
idx_t const nrows,
idx_t const ncols)
{
matrix_t * mat = mat_alloc(nrows, ncols);
val_t * const vals = mat->vals;
fill_rand(vals, nrows * ncols);
return mat;
}
void mat_free(
matrix_t * mat)
{
free(mat->vals);
free(mat);
}
matrix_t * mat_mkrow(
matrix_t const * const mat)
{
assert(mat->rowmajor == 0);
idx_t const I = mat->I;
idx_t const J = mat->J;
matrix_t * row = mat_alloc(I, J);
val_t       * const restrict rowv = row->vals;
val_t const * const restrict colv = mat->vals;
for(idx_t i=0; i < I; ++i) {
for(idx_t j=0; j < J; ++j) {
rowv[j + (i*J)] = colv[i + (j*I)];
}
}
return row;
}
matrix_t * mat_mkcol(
matrix_t const * const mat)
{
assert(mat->rowmajor == 1);
idx_t const I = mat->I;
idx_t const J = mat->J;
matrix_t * col = mat_alloc(I, J);
val_t       * const restrict colv = col->vals;
val_t const * const restrict rowv = mat->vals;
for(idx_t i=0; i < I; ++i) {
for(idx_t j=0; j < J; ++j) {
colv[i + (j*I)] = rowv[j + (i*J)];
}
}
col->rowmajor = 0;
return col;
}
spmatrix_t * spmat_alloc(
idx_t const nrows,
idx_t const ncols,
idx_t const nnz)
{
spmatrix_t * mat = (spmatrix_t*) splatt_malloc(sizeof(spmatrix_t));
mat->I = nrows;
mat->J = ncols;
mat->nnz = nnz;
mat->rowptr = (idx_t*) splatt_malloc((nrows+1) * sizeof(idx_t));
mat->colind = (idx_t*) splatt_malloc(nnz * sizeof(idx_t));
mat->vals   = (val_t*) splatt_malloc(nnz * sizeof(val_t));
return mat;
}
void spmat_free(
spmatrix_t * mat)
{
free(mat->rowptr);
free(mat->colind);
free(mat->vals);
free(mat);
}
