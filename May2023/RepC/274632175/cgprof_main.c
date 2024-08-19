#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cgprof_main.h"
#include "fptype.h"
#include "fpblas.h"
#include "task_log.h"
#include "matfprint.h"
#include "cgas_workspace.h"
#include "as_man.h"
#include "as_man.h"
#include "async_struct.h"
#include "selfsched.h"
#include "bblas_dot.h"
#include "bblas_dot_async.h" 
#include "bblas_gemm.h"
#include "bblas_axpy.h" 
#include "bblas_axpy_async.h" 
#include "bblas_copy.h"
#if USE_SPARSE
#include "hb.h"
#include "fpsblas.h"
#include "bsblas_csrmm.h"
#include "bsblas_csrmmb.h"
#endif
typedef enum {
DOT_ALPHA1, 
DOT_ALPHA2,
DOT_ALPHA3, 
} dot_t;
#if USE_SPARSE
static const char cgid[] = "cgprofs";
#else
static const char cgid[] = "cgprof";
#endif
#define PRIOR_FIRSTCP		9999999
#define PRIOR_GEMM			99999
#define PRIOR_ASYNCDOT		999999
#define PRIOR_ASYNCAXPY		9999
int CGPROF(int bm, int bn, int n, void *A, int s, void *b, void *xbuff, int *offs, double tol, int steps, void *work, unsigned long works, 
int lookahead, int async, double profile, int warmup, int cglog, int release, fp_t orth_fac, float *mat_energy, 
int is_precond, void *zbuff, cs *Acs, css **S, csn **N, int interval, int corrections) 
{
typedef struct { 
async_t sync[3];
fp_t buff;
} worklay_t;
float *residuals = calloc(steps, sizeof(float));
int bs = (n+bm-1)/bm;
int *bm_p = calloc(bs, sizeof(int));
#if USE_SPARSE
hbmat_t *Ahb = A;
fp_t *z[4] = {zbuff, zbuff+n, zbuff+2*n, zbuff+3*n};
#endif
fp_t *lxbuff = xbuff;
fp_t *x[4] = {lxbuff, lxbuff+n, lxbuff+2*n, lxbuff+3*n};
fp_t **r; 
fp_t **tmp;  
fp_t **p; 
fp_t **alpha1;
fp_t **alpha2;
worklay_t *worklay = (worklay_t*) work;
fp_t *orth = malloc(s*s*sizeof(fp_t));
fp_t porth = FP_MAXVAL;
int dupl = 4;
if ( cgas_malloc(n, s, dupl, &worklay->buff, works, &p, &tmp, &r, &alpha1, &alpha2) ) {
return -1;
}
async_t *sync =  async_init(worklay->sync, 3, n, bm, -2, 1);
async_profile(&sync[DOT_ALPHA2], profile);
async_profile(&sync[DOT_ALPHA3], profile);
asman_t *asman = asman_init(4);
int i = asman_nexti(asman);
BBLAS_COPY(PRIOR_FIRSTCP, bm, bn, n, s, b, r[i]);
#if USE_SPARSE
BSBLAS_CSRMV(PRIOR_GEMM, bm, FP_MONE, Ahb, x[i], n, FP_ONE, r[i], n);
#else
BBLAS_GEMM(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[i], n, FP_ONE, r[i], n); 					
#endif
#if USE_SPARSE
if ( is_precond ) {
BSBLAS_CHOLSOLV2(1, bm, n, S, N, r[i], z[i]);
BBLAS_COPY(PRIOR_FIRSTCP, bm, bn, n, s, z[i], p[i]);
BBLAS_DOT(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
} else {
BBLAS_COPY(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]);
BBLAS_DOT(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
}
#else
BBLAS_COPY(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]); 				
BBLAS_DOT(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);            	
#endif
int iprev = 0;
int aheadc = 0;
async_stat_t status = STAT_SYNC;
asman_update(asman, 0, 0);
int is_converged = 0;
int ncorrection = -1;
int k; 
for ( k=0; k<steps; ++k ) {
if ( interval > 0 ) {
if ( ncorrection == corrections )
ncorrection = -1;
else
ncorrection += 1;
if ( k % interval == 0) {
ncorrection = 0;
}
}
i = asman_nexti(asman);
if ( async && warmup<=0 && ncorrection==-1) {
#if USE_SPARSE
BSBLAS_CSRMV_PRIOR_RELEASE(&sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, mat_energy, bm, FP_ONE, Ahb, p[iprev], n, FP_NOUGHT, tmp[i], n, release, bm_p);
#else
BBLAS_GEMM_PRIOR_RELEASE(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], release, mat_energy, bm_p);
#endif
BBLAS_DOT_SCHED_ASYNC_CONCURRENT(k, &sync[DOT_ALPHA2], PRIOR_ASYNCDOT, bm, bn, n, s, tmp[i], p[iprev], alpha2[i], release, bm_p); 
BBLAS_SCAL_CPAXPY_COMB_ASYNC_CONCURRENT(k, &sync[DOT_ALPHA2], PRIOR_ASYNCAXPY, bm, bn, n, s, -1.0, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
} else {
#if USE_SPARSE
BSBLAS_CSRMV_PRIOR(&sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, mat_energy, bm, FP_ONE, Ahb, p[iprev], n, FP_NOUGHT, tmp[i], n);
#else
BBLAS_GEMM_PRIOR(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], mat_energy);
#endif
BBLAS_DOT(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
BBLAS_CPAXPY_COMB(bm, bn, n, s, -1.0, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
}
#if USE_SPARSE
if ( is_precond ) {
BSBLAS_CHOLSOLV2(1, bm, n, S, N, r[i], z[i]);
BBLAS_DOT(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
BBLAS_EXTM_AXPY(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], z[i], p[i]); 	
} else {
BBLAS_DOT(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
BBLAS_EXTM_AXPY(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
}
#else
BBLAS_DOT(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);             					
BBLAS_EXTM_AXPY(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
#endif
#pragma omp taskwait
BLAS_gemm(OMPSSBLAS_TRANSP, OMPSSBLAS_NTRANSP, s, s, n, FP_ONE, p[i], n, tmp[i], n, FP_NOUGHT, orth, s);
*orth = FP_ABS(*orth);
if ( cglog > 1 ) {
fprintf(stdout, "orth: %e\n", *orth);
}
if( warmup <= 0 ) {
if (isgreater(*orth, porth * orth_fac)){
break;
}
}
if ( cglog > 2 ) {
if( async && warmup <= 0 ) {
prof_dump(&(sync[DOT_ALPHA2].prof));
prof_dump(&(sync[DOT_ALPHA3].prof));
} else {
prof_dump_fake(&(sync[DOT_ALPHA2].prof), 1);
prof_dump_fake(&(sync[DOT_ALPHA3].prof), 1);
}
}
porth = *orth;
status = asman_sbreak(k, i, asman, alpha1, &sync[DOT_ALPHA1], 0, tol, &residuals[k]);
if ( status == STAT_CONVERGED ) {
is_converged = 1;
break;	
} 
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
if ( warmup > 0 ) {
warmup--;
}
}
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*alpha1[iprev])
asman_sbreak(k-1, i, asman, alpha1, &sync[DOT_ALPHA1], 1, tol, &residuals[k]);
}
*offs = asman_best(asman);
asman_fini(asman);
#pragma omp taskwait
free(p);
if ( cglog > 0 ) {
FILE *logr = fopen("cgprof_residual.log", "a");
int kk;
for (kk = 0; kk < k; kk++ ) {
fprintf(logr, "%d: %e\n", kk, residuals[kk]);
}
fprintf(logr, "ITREF iteration ends\n");
fclose(logr);
}
free(residuals);
free(bm_p);
k = is_converged ? -1 : k;
return k;
}
