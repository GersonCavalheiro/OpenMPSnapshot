#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cgmod1_main.h"
#include "fptype.h"
#include "fpblas.h"
#include "task_log.h"
#include "matfprint.h"
#include "cgmod1_workspace.h"
#include "as_man.h"
#include "bblas_dot.h"
#include "bblas_dot_async.h" 
#include "bblas_gemm.h"
#include "bblas_axpy.h" 
#include "bblas_axpy_async.h" 
#include "bblas_copy.h"
#if USE_SPARSE
#include "bsblas_csrmmb.h"
#include "hb.h"
#endif
typedef enum {
DOT_ALPHA1, 
DOT_ALPHA2,
DOT_ALPHA3, 
} dot_t;
#if USE_SPARSE
static const char cgid[] = "cgmod1s";
#else
static const char cgid[] = "cgmod1";
#endif
#define PRIOR_FIRSTCP		9999999
#define PRIOR_GEMM			99999
#define PRIOR_ASYNCDOT		999999
#define PRIOR_ASYNCAXPY		9999
int CGMOD1(int bm, int bn, int n, void *A, int s, void *b, void *xbuff, int *offs, double tol, int steps, void *work, unsigned long works, int lookahead, int async, double profile, int warmup, int cglog, int release) 
{
typedef struct { 
async_t sync[3];
fp_t buff;
} worklay_t;
fp_t *lxbuff = xbuff;
fp_t *x[4] = {lxbuff, lxbuff+n, lxbuff+2*n, lxbuff+3*n};
fp_t **r; 
fp_t **p;  			
fp_t **v; 			
fp_t **z; 			
fp_t **gamma; 		
fp_t **sigma;		
fp_t **delta;		
worklay_t *worklay = (worklay_t*) work;
int dupl = 4;
if ( cgmod1_malloc(n, s, dupl, &worklay->buff, works, &p, &v, &r, &z, &gamma, &sigma, &delta) ) {
return -1;
}
async_t *sync =  async_init(worklay->sync, 3, n, bm, -2, 0);
async_profile(&sync[DOT_ALPHA1], profile);
asman_t *asman = asman_init(4);
if ( cglog ) {
async_log(&sync[DOT_ALPHA1], 256, cgid);
async_log(&sync[DOT_ALPHA2], 256, cgid);
}
unsigned long elapsed = 0;
int broke = 0;
int iprev = 0;
int aheadc = 0;
async_stat_t status = STAT_SYNC;
int i = asman_nexti(asman);
BBLAS_COPY(1, bm, bn, n, s, b, r[i]);        									
#if USE_SPARSE
BSBLAS_CSRMMB(bm, s, FP_MONE, A, x[i], n, FP_ONE, r[i], n);
#else
BBLAS_GEMM(1, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[i], n, FP_ONE, r[i], n); 					
#endif
BBLAS_DOT(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], gamma[i]);            	
BBLAS_COPY(1, bm, bn, n, s, r[i], p[i]);        									
#if USE_SPARSE
BSBLAS_CSRMMB(bm, s, FP_ONE, A, p[i], n, FP_NOUGHT, v[i], n);
#else
BBLAS_GEMM(1, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_ONE, A, n, p[i], n, FP_NOUGHT, v[i], n);    		
#endif
BBLAS_DOT(-1, &sync[DOT_ALPHA2], bm, bn, n, s, p[i], v[i], sigma[i]);            	
BBLAS_CPAXPY_COMB(bm, bn, n, s, FP_MONE, gamma[i], sigma[i], v[i], p[i], r[i], x[i], r[i], x[i]);    	
if ( ! lookahead ) {
#pragma omp taskwait 
status = ASMAN_BREAK(-1, i,  asman, gamma, &sync[DOT_ALPHA1], 1, tol);
} else {
status = STAT_AHEAD;
}
int k; 
for ( k=0; k<steps; ++k ) {
i = asman_nexti(asman);
if ( status == STAT_AHEAD ) {
int prevk = k - 1;
#if USE_SPARSE
status = BSBLAS_CSRMMB_PRED(bm, s, FP_ONE, A, r[iprev], n, FP_NOUGHT, z[i], n,\
prevk, iprev, iback, &sync[DOT_ALPHA1], gamma, tol);
#else
status = bblas_sgemm_profpred(k, &sync[DOT_ALPHA1], &sync[DOT_ALPHA2], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, r[iprev], FP_NOUGHT, z[i],\
prevk, iprev, asman, &sync[DOT_ALPHA1], gamma, tol);
#endif
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*gamma[iprev])
status = ASMAN_BREAK(prevk, iprev, asman,  gamma, &sync[DOT_ALPHA1], 1, tol);
}
if ( status == STAT_BROKEN ) {
break;
}
} else {
#if USE_SPARSE
BSBLAS_CSRMMB(bm, s, FP_ONE, A, r[iprev], n, FP_NOUGHT, z[i], n);
#else
bblas_sgemm_prior(k, &sync[DOT_ALPHA1], &sync[DOT_ALPHA2], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, r[iprev], FP_NOUGHT, z[i]); 
#endif
}
if ( async && warmup<=0 ) {
BBLAS_DOT3_ASYNC(k, &sync[DOT_ALPHA1], PRIOR_ASYNCDOT, bm, bn, n, s, r[iprev], z[i], gamma[i], delta[i], release); 		
BBLAS_CPAXPY_COMB4_ASYNC(k, &sync[DOT_ALPHA1], PRIOR_ASYNCAXPY, bm, bn, n, s, gamma[i], gamma[iprev], delta[i], sigma[iprev], p[iprev], v[iprev], x[iprev], r[iprev], z[i],\
sigma[i], p[i], v[i], x[i], r[i]); 
} else {
BBLAS_DOT3(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[iprev], z[i], gamma[i], delta[i]); 		
BBLAS_CPAXPY_COMB4(bm, bn, n, s, gamma[i], gamma[iprev], delta[i], sigma[iprev], p[iprev], v[iprev], x[iprev], r[iprev], z[i],\
sigma[i], p[i], v[i], x[i], r[i]); 
}
if ( ! lookahead ) {
#pragma omp taskwait
}
if( warmup <= 0 ) {
prof_dump(&(sync[DOT_ALPHA1].prof));
}
status = ASMAN_BREAK(k, i, asman, gamma, &sync[DOT_ALPHA1], 0, tol);
if ( status == STAT_BROKEN ) {
break;	
} 
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
--warmup;
}
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*gamma[iprev])
ASMAN_BREAK(k-1, iprev, asman, gamma, &sync[DOT_ALPHA1], 1, tol);
}
*offs = asman_best(asman);
asman_fini(asman);
free(p);
#pragma omp taskwait
return k;
} 
