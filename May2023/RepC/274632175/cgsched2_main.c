#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cgsched_main.h"
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
typedef enum {
DOT_ALPHA1, 
DOT_ALPHA2,
DOT_ALPHA3, 
} dot_t;
#define PRIOR_FIRSTCP		9999999
#define PRIOR_GEMM			99999
#define PRIOR_ASYNCDOT		999999
#define PRIOR_ASYNCAXPY		9999
int scgsched(int bm, int bn, int n, void *A, int s, float *b, float *xbuff, int *offs, float *xstar, float tol, int steps, float *work, unsigned long works, int lookahead, int async, float profile, int warmup) 
{
float *x[4] = {xbuff, xbuff+n, xbuff+2*n, xbuff+3*n};
float **r; 
float **tmp;  
float **p; 
float **alpha1;
float **alpha2;
int bc = ( n + bm - 1 ) / bm;
int dupl = 4;
if ( cgas_malloc(n, s, dupl, work, works, &p, &tmp, &r, &alpha1, &alpha2) ) {
return -1;
}
selfsched_t *psched = sched_malloc(-2, bc);
async_t *sync =  async_init(3, n, bm, -2, 1, profile);
cgas_man_init(4);
#if 0
async_log(&sync[DOT_ALPHA1], 256, "cgsched");
async_log(&sync[DOT_ALPHA2], 1024, "cgsched");
async_log(&sync[DOT_ALPHA3], 1024, "cgsched");
#endif
bblas_scopy(PRIOR_FIRSTCP, bm, bn, n, s, b, r[0]);	                     						
bblas_sgemm(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[0], n, FP_ONE, r[0], n); 					
bblas_scopy_sched(&sync[DOT_ALPHA2], -1, PRIOR_FIRSTCP, bm, bn, n, s, r[0], psched, p[0]); 				
bblas_sdot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[0], r[0], alpha1[0]);            	
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
if ( status == STAT_AHEAD ) {
int prevk = k - 1;
status = bblas_sgemm_pred_sched(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, 1.0, A, psched, p[iprev], FP_NOUGHT, tmp[i],\
prevk, iprev, iback, &sync[DOT_ALPHA1], alpha1, tol);
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*alpha1[iprev])
status = cgas_man_break(prevk, iprev, iback, alpha1, &sync[DOT_ALPHA1], 1, tol);
}
if ( status == STAT_BROKEN || status == STAT_CONVERGED ) {
break;
}
} else {
#if 0
BBLAS_GEMM_PRIOR(bm, bn, bm, n, s, n, 1.0, A, p[iprev], FP_NOUGHT, tmp[i],\
k, iprev, iback, &sync[DOT_ALPHA1], alpha1, tol);    		
#endif
bblas_sgemm_prior(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, 1.0, A, p[iprev], FP_NOUGHT, tmp[i]);
}
if ( async && warmup<=0 ) {
bblas_sdot_sched_async(k, &sync[DOT_ALPHA2], PRIOR_ASYNCDOT, bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
bblas_scal_scpaxpy_comb_async(k, &sync[DOT_ALPHA2], PRIOR_ASYNCAXPY, bm, bn, n, s, -1.0, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
} else {
bblas_sdot_prof(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
bblas_scpaxpy_comb(bm, bn, n, s, -1.0, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
}
bblas_sdot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);             					
bblas_extm_saxpy_sched(k, PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], psched, p[iprev], r[i], p[i]); 	
if ( ! lookahead ) {
#pragma omp taskwait
}
status = cgas_man_break(k, i, iprev, alpha1, &sync[DOT_ALPHA1], 0, tol);
if ( status == STAT_BROKEN || status == STAT_CONVERGED ) {
printf("breaking\n");
break;	
} 
iback = iprev;
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
--warmup;
}
if ( status == STAT_AHEAD ) {
#pragma omp taskwait on (*alpha1[iprev])
cgas_man_break(k-1, i, iback, alpha1, &sync[DOT_ALPHA1], 1, tol);
}
#pragma omp taskwait
*offs = cgas_man_best();
async_fini(sync, 3);
cgas_man_fini();
sched_free(psched);
free(p);
return k;
} 
