#ifndef __BSBLAS_CSRMM_H__
#define __BSBLAS_CSRMM_H__
#include "fptype.h"
#include "fpblas.h"
#include "hb.h"
#include "task_csrmm.h"
#include "matfprint.h"
#include "selfsched.h"
#include "array.h"
#include "async_struct.h"
#include "assert.h"
#include "scsparse.h"
#ifdef SINGLE_PRECISION
#define BSBLAS_CSRMM				bsblas_scsrmm
#define BSBLAS_CSRMV				bsblas_scsrmv
#define HBSBLAS_CSRMV				hbsblas_scsrmv
#define BSBLAS_CSRMV_PRIOR			bsblas_scsrmv_prior
#define BSBLAS_CSRMV_PRIOR_RELEASE	bsblas_scsrmv_prior_release
#define BSBLAS_CSRSM				bsblas_scsrsm
#define BSBLAS_CHOLSOLV2			bsblas_scholsolv2
#define BSBLAS_CSRMV_RELEASE		bsblas_scsrmv_release
#endif
#ifdef DOUBLE_PRECISION
#define BSBLAS_CSRMM				bsblas_dcsrmm
#define BSBLAS_CSRMV				bsblas_dcsrmv
#define HBSBLAS_CSRMV				hbsblas_dcsrmv
#define BSBLAS_CSRMV_PRIOR			bsblas_dcsrmv_prior
#define BSBLAS_CSRMV_PRIOR_RELEASE	bsblas_dcsrmv_prior_release
#define BSBLAS_CSRSM				bsblas_dcsrsm
#define BSBLAS_CHOLSOLV2			bsblas_dcholsolv2
#define BSBLAS_CSRMV_RELEASE		bsblas_dcsrmv_release
#endif
static inline void __attribute__((always_inline)) bsblas_scsrmm(int b, int n, float alpha, hbmat_t *Ahbh, float *B, int ldb, float beta, float *C, int ldc, selfsched_t *sched) 
{
int M = Ahbh->m;
int N = Ahbh->n;
int *vptr = Ahbh->vptr;
int *vpos = Ahbh->vpos;
hbmat_t **vval = Ahbh->vval;
int offs = vptr[0] == 0 ? 0 : 1;
int cmaj = 1;
int I;
for ( I = 0; I < M; ++I ) {
int is = vptr[I] - offs;  
int bcnt = vptr[I+1] - vptr[I];
int ib = vval[is]->m;
int J;
for ( J = 0; J < n; J+=b ) {
int boffs = cmaj ? J * ldb : 0 ;
int coffs = cmaj ? J * ldb + I*b : 0 ;
int jleft = n - J;
int c = jleft < b ? jleft : b;
float *bC = &(C[coffs]);
float *bB = &(B[boffs]);
task_scsrmm_sched(bcnt, ib, c, b, alpha, &vptr[I], &vpos[is], &vval[is], bB, ldb, beta, bC, ldc, sched);
}
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmm(int b, int n, float alpha, hbmat_t *Ahbh, float *B, int ldb, float beta, float *C, int ldc, selfsched_t *sched) 
{
int M = Ahbh->m;
int N = Ahbh->n;
int *vptr = Ahbh->vptr;
int *vpos = Ahbh->vpos;
hbmat_t **vval = Ahbh->vval;
int offs = vptr[0] == 0 ? 0 : 1;
int cmaj = 1;
int I;
for ( I = 0; I < M; ++I ) {
int is = vptr[I] - offs;  
int bcnt = vptr[I+1] - vptr[I];
int ib = vval[is]->m;
int J;
for ( J = 0; J < n; J+=b ) {
int boffs = cmaj ? J * ldb : 0 ;
int coffs = cmaj ? J * ldb + I*b : 0 ;
int jleft = n - J;
int c = jleft < b ? jleft : b;
double *bC = &(C[coffs]);
double *bB = &(B[boffs]);
task_dcsrmm_sched(bcnt, ib, c, b, alpha, &vptr[I], &vpos[is], &vval[is], bB, ldb, beta, bC, ldc, sched);
}
}
}
static inline void __attribute__((always_inline)) hbsblas_scsrmv(int p, int b, float alpha, hbmat_t *Ahbh, float *B, float beta, float *C)
{
int M = Ahbh->m;
int N = Ahbh->n;
int *vptr = Ahbh->vptr;
int *vpos = Ahbh->vpos;
hbmat_t **vval = Ahbh->vval;
int offs = vptr[0] == 0 ? 0 : 1; 
int cmaj = 1;
char *trans = "N";
char *matdescra = "GLNC";
int I;
for ( I = 0; I < M; ++I ) {
float *Cptr = &C[I*b];
int first = 1;
int J;
for ( J = vptr[I]; J < vptr[I+1]; J++ ) {
hbmat_t *A = vval[J];
int icol = vpos[J];
float *Bptr = &B[icol*b];
float *avval = A->vval;
int *avpos = A->vpos;
int *avptr = A->vptr;
int m = A->m;
int n = A->n;
if ( first ) {
#pragma omp task in([n]Bptr) out([m]Cptr) no_copy_deps label(csrmv_hbh) priority(p)
mkl_scsrmv(trans, &m, &n, &alpha, matdescra, avval, avpos, avptr, avptr+1, Bptr, &beta, Cptr);
first = 0;
} else {
#pragma omp task in([n]Bptr) out([m]Cptr) no_copy_deps label(csrmv_hbh) priority(p)
mkl_scsrmv(trans, &m, &n, &alpha, matdescra, avval, avpos, avptr, avptr+1, Bptr, &FP_ONE, Cptr);
}
}
}
}
static inline void __attribute__((always_inline)) hbsblas_dcsrmv(int p, int b, double alpha, hbmat_t *Ahbh, double *B, double beta, double *C) 
{
int M = Ahbh->m;
int N = Ahbh->n;
int *vptr = Ahbh->vptr;
int *vpos = Ahbh->vpos;
hbmat_t **vval = Ahbh->vval;
int offs = vptr[0] == 0 ? 0 : 1; 
int cmaj = 1;
char *trans = "N";
char *matdescra = "GLNC";
int I;
for ( I = 0; I < M; ++I ) {
double *Cptr = &C[I*b];
int first = 1;
int J;
for ( J = vptr[I]; J < vptr[I+1]; J++ ) {
hbmat_t *A = vval[J];
int icol = vpos[J];
double *Bptr = &B[icol*b];
double *avval = A->vval;
int *avpos = A->vpos;
int *avptr = A->vptr;
int m = A->m;
int n = A->n;
if ( first ) {
#pragma omp task in([n]Bptr) out([m]Cptr) no_copy_deps label(csrmv_hbh) priority(p)
mkl_dcsrmv(trans, &m, &n, &alpha, matdescra, avval, avpos, avptr, avptr+1, Bptr, &beta, Cptr);
first = 0;
} else {
#pragma omp task in([n]Bptr) out([m]Cptr) no_copy_deps label(csrmv_hbh) priority(p)
mkl_dcsrmv(trans, &m, &n, &alpha, matdescra, avval, avpos, avptr, avptr+1, Bptr, &FP_ONE, Cptr);
}
}
}
}
static inline void __attribute__((always_inline)) bsblas_scsrmv(int p, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C)
{
int m = Ahb->m; int n = Ahb->n;
assert(m==n);
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int i;
for ( i = 0; i < m; i+=b ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
task_scsrmv(p, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmv(int p, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C)
{
int m = Ahb->m; int n = Ahb->n;
assert(m==n);
int* vptr = Ahb->vptr; int* vpos = Ahb->vpos; double* vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int i;
for ( i = 0; i < m; i+=b ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
task_dcsrmv(p, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr);
}
}
#if 0
static inline void __attribute__((always_inline)) bsblas_scsrmv2(int p, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C)
{
int m = Ahb->m; int n = Ahb->n;
assert(m==n);
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int i;
for ( i = 0; i < m; i+=b ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
task_scsrmv(p, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmv2(int p, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C)
{
int m = Ahb->m; int n = Ahb->n;
assert(m==n);
int* vptr = Ahb->vptr; int* vpos = Ahb->vpos; double* vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int i;
for ( i = 0; i < m; i+=b ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
int j;
for ( j = 0; j < n; j += b ) {
}
task_dcsrmv(p, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr);
}
}
#endif
static inline void __attribute__((always_inline)) bsblas_scsrmv_prior(async_t *sync, async_t *sync2, int basep, float *mat_energy, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
int resolution = sync->prof.resolution;
int szeb = (m+b-1)/b;
float M = mat_energy[szeb];
float scal = (1 / M * resolution);
float len = fabs(mat_energy[idx] * scal);
int childp = basep + (int) len;
task_scsrmv_prior(childp, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmv_prior(async_t *sync, async_t *sync2, int basep, double *mat_energy, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; double *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
int resolution = sync->prof.resolution;
int szeb = (m+b-1)/b;
double M = mat_energy[szeb];
double scal = (1 / M * resolution);
double len = fabs(mat_energy[idx] * scal);
int childp = basep + (int) len;
task_dcsrmv_prior(childp, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr);
}
}
static inline void __attribute__((always_inline)) bsblas_scsrmv_prior_release(async_t *sync, async_t *sync2, int basep, float *mat_energy, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C, int release, int *bitmap)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
int resolution = sync->prof.resolution;
int szeb = (m+b-1)/b;
float M = mat_energy[szeb];
float scal = (1 / M * resolution);
float len = fabs(mat_energy[idx] * scal);
int childp = basep + (int) len;
task_scsrmv_release(sync2, idx, childp, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, release, bitmap);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmv_prior_release(async_t *sync, async_t *sync2, int basep, double *mat_energy, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C, int release, int *bitmap)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; double *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
int resolution = sync->prof.resolution;
int szeb = (m+b-1)/b;
double M = mat_energy[szeb];
double scal = (1 / M * resolution);
double len = fabs(mat_energy[idx] * scal);
int childp = basep + (int) len;
task_dcsrmv_release(sync2, idx, childp, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, release, bitmap);
}
}
static inline void __attribute__((always_inline)) bsblas_scsrmv_prior_switch(async_t *sync, async_t *sync2, int basep, float *mat_energy, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C, int release_id, int *bitmap)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
int resolution = sync->prof.resolution;
int szeb = (m+b-1)/b;
float M = mat_energy[szeb];
float scal = (1 / M * resolution);
float len = fabs(mat_energy[idx] * scal);
int childp = basep + (int) len;
int on = ( idx == release_id ) ? 0 : 1;
task_scsrmv_switch(sync2, idx, childp, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, release_id, bitmap, on);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmv_prior_switch(async_t *sync, async_t *sync2, int basep, double *mat_energy, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C, int release_id, int *bitmap)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; double *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
int resolution = sync->prof.resolution;
int szeb = (m+b-1)/b;
double M = mat_energy[szeb];
double scal = (1 / M * resolution);
double len = fabs(mat_energy[idx] * scal);
int childp = basep + (int) len;
int on = ( idx == release_id ) ? 0 : 1;
task_dcsrmv_switch(sync2, idx, childp, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, release_id, bitmap, on);
}
}
static inline void __attribute__((always_inline)) bsblas_scsrsm(int p, int m, int n, int b, hbmat_t *A, float *B, float *x)
{
char *trans = "N";
char *matdescra = "TLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i+=b, idx++) {
float *xptr = &x[i];
float *bptr = &B[i];
hbmat_t *Ahb = &A[idx];
task_scsrsm(p, trans, Ahb->m, n, FP_ONE, matdescra, Ahb->vval, Ahb->vpos, Ahb->vptr, bptr, n, xptr, n);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrsm(int p, int m, int n, int b, hbmat_t *A, double *B, double *x)
{
char *trans = "N";
char *matdescra = "TLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i+=b, idx++) {
double *xptr = &x[i];
double *bptr = &B[i];
hbmat_t *Ahb = &A[idx];
task_dcsrsm(p, trans, Ahb->m, n, FP_ONE, matdescra, Ahb->vval, Ahb->vpos, Ahb->vptr, bptr, n, xptr, n);
}
}
static inline void __attribute__((always_inline)) bsblas_scholsolv2(int p, int b, int m, css **S, csn **N, float *B, float *x)
{
int idx;
int i;
for ( i = 0, idx = 0; i < m; i+=b, idx++) {
int bs = b < m-i ? b : m-i;
css *sptr = S[idx];
csn *nptr = N[idx];
float *bptr = &B[i];
float *xptr = &x[i];
#pragma omp task in([bs]bptr) out([bs]xptr) label(scholsolv2) no_copy_deps
{
cs_cholsol2_double(bs, sptr, nptr, bptr, xptr);
}
}
}
static inline void __attribute__((always_inline)) bsblas_dcholsolv2(int p, int b, int m, css **S, csn **N, double *B, double *x)
{
int idx;
int i;
for ( i = 0, idx = 0; i < m; i+=b, idx++) {
int bs = b < m-i ? b : m-i;
css *sptr = S[idx];
csn *nptr = N[idx];
float *bptr = &B[i];
float *xptr = &x[i];
#pragma omp task in([bs]bptr) out([bs]xptr) label(scholsolv2)
{
cs_cholsol2_double(bs, sptr, nptr, bptr, xptr);
}
}
}
static inline void __attribute__((always_inline)) bsblas_scg_comb(int id, async_t *sync, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C, float *result)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
task_scsrmv_comb(sync, id, idx, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, result, 1);
}
}
static inline void __attribute__((always_inline)) bsblas_dcg_comb(int id, async_t *sync, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C, double *result)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; double *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
task_dcsrmv_comb(sync, id, idx, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, result, 1);
}
}
static inline void __attribute__((always_inline)) bsblas_scg_comb_switch(int id, async_t *sync, int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C, float *result, int release_id)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
int on = ( idx == release_id ) ? 0 : 1;
task_scsrmv_comb(sync, id, idx, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, result, on);
}
}
static inline void __attribute__((always_inline)) bsblas_dcg_comb_switch(int id, async_t *sync, int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C, double *result, int release_id)
{
int m = Ahb->m; int n = Ahb->n;
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; double *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx;
int i;
for ( i = 0, idx = 0; i < m; i += b, idx++ ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
int on = ( idx == release_id ) ? 0 : 1;
task_dcsrmv_comb(sync, id, idx, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, result, on);
}
}
static inline void __attribute__((always_inline)) bsblas_scsrmv_switch(int b, float alpha, hbmat_t *Ahb, float *B, float beta, float *C, int release_id)
{
int m = Ahb->m; int n = Ahb->n;
assert(m==n);
int *vptr = Ahb->vptr; int *vpos = Ahb->vpos; float *vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx = 0;
int i;
for ( i = 0; i < m; i+=b, idx++ ) {
int bs = b < m-i ? b : m-i;
float *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
float *vval1 = &vval[*vptr1];
int on = ( idx == release_id ) ? 0 : 1;
task_scsrmv_switch0(idx, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, on);
}
}
static inline void __attribute__((always_inline)) bsblas_dcsrmv_switch(int b, double alpha, hbmat_t *Ahb, double *B, double beta, double *C, int release_id)
{
int m = Ahb->m; int n = Ahb->n;
assert(m==n);
int* vptr = Ahb->vptr; int* vpos = Ahb->vpos; double* vval = Ahb->vval;
char *trans = "N";
char *matdescra = "GLNC";
int idx = 0;
int i;
for ( i = 0; i < m; i+=b, idx++ ) {
int bs = b < m-i ? b : m-i;
double *Cptr = &C[i];
int *vptr1 = &vptr[i];
int *vpos1 = &vpos[*vptr1];
double *vval1 = &vval[*vptr1];
int on = ( idx == release_id ) ? 0 : 1;
task_dcsrmv_switch0(idx, trans, bs, m, alpha, matdescra, vval1, vpos1, vptr1, B, beta, Cptr, on);
}
}
#endif 
