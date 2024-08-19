#include <omp.h>
#include "task_csrmm.h"
#include "fptype.h"
#include "fpblas.h"
#include "fpsblas.h"
#include "selfsched.h"
#include "task_csrmmb.h"
#include "matfprint.h"
#include "async_struct.h"
#ifdef SINGLE_PRECISION
#define _t_csrmm_sched		task_scsrmm_sched
#define _t_csrmv			task_scsrmv
#define _t_csrmv_prior		task_scsrmv_prior
#define _t_csrmv_release	task_scsrmv_release
#define _t_csrmv_switch		task_scsrmv_switch
#define _t_csrsm			task_scsrsm
#define _t_csrmv_comb		task_scsrmv_comb
#define _t_csrmv_switch0	task_scsrmv_switch0
#else
#define _t_csrmm_sched		task_dcsrmm_sched
#define _t_csrmv			task_dcsrmv
#define _t_csrmv_prior		task_dcsrmv_prior
#define _t_csrmv_release	task_dcsrmv_release
#define _t_csrmv_switch		task_dcsrmv_switch
#define _t_csrsm			task_dcsrsm
#define _t_csrmv_comb		task_dcsrmv_comb
#define _t_csrmv_switch0	task_dcsrmv_switch0
#endif
void _t_csrmm_sched(int cnt, int b, int c, int d, fp_t alpha, int *vptr, int *vpos, hbmat_t **vval, fp_t *B, int ldb, fp_t beta, fp_t *C, int ldc, selfsched_t *sched)
{
if ( cnt > 0 ) {
hbmat_t *Ahb = vval[0];
int offs = vpos[0];	
TASK_CSRMMB(c, alpha, "GLNF", Ahb, B + offs * d, ldb, beta, C, ldc, 0, sched);
int k;
for ( k = 1; k < cnt-1; ++k ) {
hbmat_t *Ahb = vval[k];
int offs = vpos[k];	
TASK_CSRMMB(c, alpha, "GLNF", Ahb, B + offs * d, ldb, FP_ONE, C, ldc, k, sched);
}
int lst = cnt - 1;
if ( lst > 0 ) {
hbmat_t *Ahb = vval[lst];
int offs = vpos[lst];	
TASK_CSRMMB(c, alpha, "GLNF", Ahb, B + offs * d, ldb, FP_ONE, C, ldc, lst, sched);
}
}
#pragma omp taskwait
}
void _t_csrmv(int p, char *trans, int bs, int m, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *B, fp_t beta, fp_t *C)
{
SBLAS_csrmv(trans, bs, m, alpha, matdescra, vval, vpos, vptr, vptr+1, B, beta, C);
}
void _t_csrmv_prior(int p, char *trans, int bs, int m, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *B, fp_t beta, fp_t *C)
{
SBLAS_csrmv(trans, bs, m, alpha, matdescra, vval, vpos, vptr, vptr+1, B, beta, C);
}
void _t_csrmv_release(async_t *sync, int idx, int p, char *trans, int bs, int m, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *B, fp_t beta, fp_t *C, int release, int *bitmap)
{
int i;
for ( i = 0; i < bs; i++ ) {
C[i] = 0.0;
}
int pcompl = sync->pcompl;
int pcnt;
#pragma omp critical
{
pcnt = sync->pcnt++;
if ( sync->pcnt == sync->pcompl ) {
sync->pcnt = 0;
}
prof_add(&sync->prof, idx, 1.0);
if ( pcnt >= pcompl - release ) {
prof_reg(&sync->prof, idx, 0);
} else {
prof_reg(&sync->prof, idx, 1);
}
}
if (  pcnt < pcompl - release ) {
bitmap[idx] = 1;
SBLAS_csrmv(trans, bs, m, alpha, matdescra, vval, vpos, vptr, vptr+1, B, beta, C);
} else {
bitmap[idx] = 0;
}
}
void _t_csrmv_switch(async_t *sync, int idx, int p, char *trans, int bs, int m, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *B, fp_t beta, fp_t *C, int release, int *bitmap, int on)
{
if ( on ) {
bitmap[idx] = 1;
SBLAS_csrmv(trans, bs, m, alpha, matdescra, vval, vpos, vptr, vptr+1, B, beta, C);
} else {
bitmap[idx] = 0;
fp_t *Bptr = &B[bs*idx];
BLAS_copy(bs, Bptr, 1, C, 1);
}
}
void _t_csrsm(int p, char *trans, int m, int n, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *b, int ldb, fp_t *x, int ldx)
{
SBLAS_csrsm(trans, m, n, alpha, matdescra, vval, vpos, vptr, vptr+1, b, ldb, x, ldx);
}
void _t_csrmv_comb(async_t *sync, int id, int idx, char *trans, int bs, int m, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *B, fp_t beta, fp_t *C, fp_t *result, int on)
{
fp_t *Bptr = &B[bs*idx];
if ( on ) {
SBLAS_csrmv(trans, bs, m, alpha, matdescra, vval, vpos, vptr, vptr+1, B, beta, C);
} else {
BLAS_copy(bs, Bptr, 1, C, 1);
}
fp_t local_result = BLAS_dot(bs, Bptr, i_one, C, i_one);
#pragma omp critical
{
if ( sync->create == id ) {
BLAS_axpy(1, FP_ONE, &local_result, i_one, result, i_one);
} else {
BLAS_copy(1, &local_result, i_one, result, i_one);
sync->create = id;
}
}
}
void _t_csrmv_switch0(int idx, char *trans, int bs, int m, fp_t alpha, char *matdescra, fp_t *vval, int *vpos, int *vptr, fp_t *B, fp_t beta, fp_t *C, int on)
{
if ( on ) {
SBLAS_csrmv(trans, bs, m, alpha, matdescra, vval, vpos, vptr, vptr+1, B, beta, C);
} else {
fp_t *Bptr = &B[bs*idx];
BLAS_copy(bs, Bptr, 1, C, 1);
}
}
