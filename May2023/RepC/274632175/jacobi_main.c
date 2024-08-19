#include "jacobi_main.h"
#include "hbext.h"
#include "fplapack.h"
#include "bblas_dot.h"
#include "tasks_trsm_csr.h"
#include "tasks_gemv_csr.h"
#include "tasks_potrf_csr.h"
#include "bsblas_gemv_csr.h"
#include "densutil.h"
#include "task_log.h"
#include "async_struct.h"
#ifdef SINGLE_PRECISION
#define jacobi_main_csr		sjacobi_main_csr
#else
#define jacobi_main_csr		djacobi_main_csr
#endif
unsigned int jacobi_main_csr(hbmat_t *Acsr, fp_t *x, fp_t *b, int bs, int max_iter, hbmat_t **diagL, int lookahead, fp_t threshold, fp_t *work, int* res_p) 
{
int M = Acsr->m; 
int N = Acsr->n;
int *vdiag = Acsr->vdiag;
int *vptr = Acsr->vptr; 
int *vpos = Acsr->vpos; 
hbmat_t **vval = Acsr->vval;
int dim = Acsr->orig->m;
fp_t *cnorm = work;
fp_t *x0 = &work[dim];
fp_t *cdot = &work[2*dim];
fp_t *vtmp_s = x;
fp_t *vtmp_d = x + dim;
fp_t n_b = VECTOR_2NORM(b, dim);
async_t *dotsync = async_init(1, dim, bs, -1, 0, 0);
async_log(dotsync, 256, "jacobi");
async_stat_t status = STAT_SYNC;
int aheadc = 0;
int prevk;
int k;
for ( k = 0; k < max_iter; ++k) {
if ( status == STAT_AHEAD ) {
status = BSBLAS_NODIAG_CP_GEMV_CSR_PRED(k, Acsr, bs, FP_MONE, vtmp_s, FP_ONE, b, vtmp_d, threshold, cdot, dotsync, n_b);
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*cdot)
status = ASYNC_CONV(k, threshold, cdot, n_b, dotsync, 1);
}
if ( status == STAT_CONVERGED ) {
break;
}
} else {
BSBLAS_NODIAG_CP_GEMV_CSR(Acsr, bs, FP_MONE, vtmp_s, FP_ONE, b, vtmp_d);
}
int I;
for ( I = 0; I < N; ++I) {
TASK_TRSM_CSR("N", FP_ONE, "TLNC", diagL[I], &(vtmp_d[I * bs]), &(x0[I*bs]));
TASK_TRSM_CSR("T", FP_ONE, "TLNC", diagL[I], &(x0[I*bs]), &(vtmp_d[I*bs]));
}
fp_t *exch = vtmp_s;
vtmp_s = vtmp_d;
vtmp_d = exch;
BSBLAS_CP_GEMV_CSR(Acsr, bs, FP_MONE, vtmp_d, FP_ONE, b, cnorm);
BBLAS_DOT(k, dotsync, bs, 1, dim, 1, cnorm, cnorm, cdot);
if (!lookahead) {
#pragma omp taskwait
}
status = ASYNC_CONV(k, threshold, cdot, n_b, dotsync, 0);
if ( status == STAT_CONVERGED ){
break;
}
prevk = k;
aheadc += status == STAT_AHEAD ? 1 : 0;
}
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*cdot)
ASYNC_CONV(k, threshold, cdot, n_b, dotsync, 1);
}
async_fini(dotsync, 1);
printf("GS: aheadc %i\n", aheadc);
*res_p = k%2 ? dim : 0;
return k;
}
