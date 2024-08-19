#include "gauss-seidel_main.h"
#include "hbext.h"
#include "fplapack.h"
#include "bblas_dot.h"
#include "bblas_copy.h"
#include "tasks_trsm_csr.h"
#include "tasks_gemv_csr.h"
#include "tasks_potrf_csr.h"
#include "bsblas_gemv_csr.h"
#include "densutil.h"
#include "task_log.h"
#include "async_struct.h"
#ifdef SINGLE_PRECISION
#define TASK_COPY		task_scopy
#define _gs_main_csr	sgs_main_csr
#else
#define TASK_COPY		task_dcopy
#define _gs_main_csr	dgs_main_csr
#endif
int _gs_main_csr(hbmat_t *Acsr, fp_t *x, fp_t *b, int bs, int max_iter, hbmat_t **diagL, int lookahead, fp_t threshold, fp_t *work) 
{
int M = Acsr->m; 
int N = Acsr->n;
int *vptr = Acsr->vptr; 
int *vpos = Acsr->vpos; 
hbmat_t **vval = Acsr->vval;
int n = Acsr->orig->m;
fp_t *cnorm = work;
fp_t *x0 = &work[n];
fp_t *cdot = &work[2*n];
fp_t *vtmp_s = x;
fp_t *vtmp_d = x + n;
fp_t n_b = VECTOR_2NORM(b, n);
async_t *dotsync = async_init(1, n, bs, -1, 0, 0);
async_log(dotsync, 256, "gs");
async_stat_t status = STAT_SYNC;
int aheadc = 0;
int prevk;
int brk = 0;
int k;
for ( k = 0; k < max_iter; ++k ) {
int I;
for ( I = 0; I < M; ++I ) {
if ( status == STAT_AHEAD ) {
int v_len = n - I * bs;
v_len = v_len >= bs ? bs : v_len;
TASK_COPY(OMPSS_PRIOR_DFLT, v_len, 1, Acsr->orig->m, 1, &(b[I*bs]), &(vtmp_s[I*bs]));
status = ASYNC_CONV(k, threshold, cdot, n_b, dotsync, 0);
if ( status == STAT_CONVERGED ) {
++brk;
break;
}
int J = vptr[I];
for ( ; J < vptr[I+1]; ++J ) {
if ( vpos[J] != I ) {
TASK_GEMV_CSR(vval[J], "N", FP_MONE, "GLNC", FP_ONE, &(vtmp_s[vpos[J] * bs]), &(vtmp_s[I * bs]));
}
}
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*cdot)
status = ASYNC_CONV(prevk, threshold, cdot, n_b, dotsync, 1);
}
if ( status == STAT_CONVERGED ) {
++brk;
break;
}
} else {
int v_len = n - I * bs;
v_len = v_len >= bs ? bs : v_len;
TASK_COPY(OMPSS_PRIOR_DFLT, v_len, 1, Acsr->orig->m, 1, &(b[I*bs]), &(vtmp_s[I*bs]));
int J = vptr[I];
for ( ; J < vptr[I+1]; ++J ) {
if ( vpos[J] != I ) {
TASK_GEMV_CSR(vval[J], "N", FP_MONE, "GLNC", FP_ONE, &(vtmp_s[vpos[J] * bs]), &(vtmp_s[I * bs]));
}
}
}
TASK_TRSM_CSR("N", FP_ONE, "TLNC", diagL[I], &(vtmp_s[I * bs]), &(x0[I*bs]));
TASK_TRSM_CSR("T", FP_ONE, "TLNC", diagL[I], &(x0[I*bs]), &(vtmp_s[I*bs]));
}
if ( brk ) {
break;
}
BBLAS_COPY(OMPSS_PRIOR_DFLT, bs, 1, n, 1, vtmp_s, vtmp_d);
BSBLAS_CP_GEMV_CSR(Acsr, bs, FP_MONE, vtmp_s, FP_ONE, b, cnorm);
BBLAS_DOT(k, dotsync, bs, 1, n, 1, cnorm, cnorm, cdot);
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
return 0;
}
