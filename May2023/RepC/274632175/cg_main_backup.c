#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cg_main.h"
#include "fptype.h"
#include "fpblas.h"
#include "task_log.h"
#include "matfprint.h"
#include "cgas_workspace.h"
#include "as_man.h"
#include "async_struct.h"
#include "bblas_dot.h"
#include "bblas_dot_async.h" 
#include "bblas_gemm.h"
#include "bblas_axpy.h" 
#include "bblas_axpy_async.h" 
#include "bblas_copy.h"
#if USE_CGS
#include "hb.h"
#include "bsblas_csrmmb.h"
#endif
typedef enum {
DOT_ALPHA1, 
DOT_ALPHA2,
DOT_ALPHA3, 
} dot_t;
#define PRIOR_ASYNCDOT		999
#define PRIOR_ASYNCAXPY		1
#if USE_CG
const char cgid[] = "cg";
#else
const char cgid[] = "cgs";
#endif
int CG(int bm, int bn, int n, void *A, int s, fp_t *b, fp_t *xbuff, int *offs, fp_t *xstar, fp_t tol, int steps, fp_t *work, unsigned long works, int lookahead, int async, fp_t profile) 
{
fp_t *x[4] = {xbuff, xbuff+n, xbuff+2*n, xbuff+3*n};
fp_t **r; 
fp_t **tmp;  
fp_t **p; 
fp_t **alpha1;
fp_t **alpha2;
int dupl = 4;
if ( cgas_malloc(n, s, dupl, work, works, &p, &tmp, &r, &alpha1, &alpha2) ) {
return -1;
}
async_t *sync =  async_init(3, n, bm, -2, 0, profile);
cgas_man_init(4);
async_log(&sync[DOT_ALPHA1], 256, cgid);
async_log(&sync[DOT_ALPHA2], 256, cgid);
BBLAS_COPY(1, bm, bn, n, s, b, r[0]);	   																						
#if USE_CG
BBLAS_GEMM(1, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[0], n, FP_ONE, r[0], n); 			
#else
BSBLAS_CSRMMB(bm, s, FP_MONE, A, x[0], n, FP_ONE, r[0], n);
#endif
BBLAS_COPY(1, bm, bn, n, s, r[0], p[0]);        																				
BBLAS_DOT(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[0], r[0], alpha1[0]);            											
int broke = 0;
int iprev = 0;
int iback = 0;
int active = 0;
int aheadc = 0;
int i;
async_stat_t status = STAT_SYNC;
int k; 
for ( k=0; k<steps; ++k ) {
i = cgas_man_nexti();
#if 0
if ( i < 0 || i > 3 ) {
printf("err: i %i\n", i);
cgas_man_repair();
fflush(0);
}
#endif
if ( status == STAT_AHEAD ) {
int prevk = k - 1;
#if USE_CG
status = BBLAS_GEMM_PRED(bm, bn, bm, n, s, n, 1.0, A, p[iprev], FP_NOUGHT, tmp[i],\
prevk, iprev, iback, &sync[DOT_ALPHA1].ready, alpha1, tol);
#else
status = BSBLAS_CSRMMB_PRED(bm, s, FP_ONE, A, p[iprev], n, FP_NOUGHT, tmp[i], n,\
prevk, iprev, iback, &sync[DOT_ALPHA1].ready, alpha1, tol);
#endif
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*alpha1[iprev])
status = cgas_man_break(prevk, iprev, iback, alpha1, &sync[DOT_ALPHA1], 1, tol);
}
if ( status == STAT_BROKEN || status == STAT_CONVERGED ) {
break;
}
} else {
#if USE_CG
BBLAS_GEMM(999, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_ONE, A, n, p[iprev], n, FP_NOUGHT, tmp[i], n);    		
#else
BSBLAS_CSRMMB(bm, s, FP_ONE, A, p[iprev], n, FP_NOUGHT, tmp[i], n);
#endif
}
if ( async ) {
BBLAS_DOT_ASYNC(k, &sync[DOT_ALPHA2], PRIOR_ASYNCDOT, bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
BBLAS_SCAL_CPAXPY_COMB_ASYNC(k, &sync[DOT_ALPHA2], PRIOR_ASYNCAXPY, bm, bn, n, s, -1.0, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
} else {
BBLAS_DOT(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
BBLAS_CPAXPY_COMB(bm, bn, n, s, -1.0, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
}
BBLAS_DOT(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);             		
BBLAS_EXTM_AXPY(bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
if ( ! lookahead ) {
#pragma omp taskwait
}
status = cgas_man_break(k, i, iprev, alpha1, &sync[DOT_ALPHA1], 0, tol);
if ( status == STAT_BROKEN || status == STAT_CONVERGED ) {
break;	
} 
iback = iprev;
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
}
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*alpha1[iprev])
cgas_man_break(k, i, iback, alpha1, &sync[DOT_ALPHA1], 1, tol);
}
*offs = cgas_man_best();
async_fini(sync, 3);
cgas_man_fini();
printf("CG: aheadc %i\n", aheadc);
free(p);
return k;
} 
