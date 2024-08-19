#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "itref_config.h"
#include "itref_setup.h"
#include "fptype.h"
#include "fpblas.h"
#include "task_log.h"
#include "as_man.h"
#include "bblas_copy.h"
#include "bblas_gemm.h"
#include "matfprint.h"
#include "bblas_convert.h"
#include "bblas_util.h"
#include "densutil.h"
#include "bblas_dot.h"
#include "bblas_dot_async.h" 
#include "bblas_gemm.h"
#include "bblas_axpy.h" 
#include "bblas_axpy_async.h" 
int n; 
int bm; 
int bn; 
int s; 
int refit; 
int it; 
double prec; 
int rep; 
double *A; 
double *rhs;  
double *x[2]; 
#include "dcsparse.h"
#if USE_SPARSE
#include "hb.h"
#include "bsblas_csrmm.h"
#include "bsblas_csrmmb.h"
void *Ahb;
void *Acsr;
void *Acsrs;
double *dvval;
void *preconditioner;
css **S;
csn **N;
cs *Acs;
#endif
double *work; 
char *rhsfname;
char *aname;
unsigned long works;
int is_precond;
int async;
double profile;
int warmup;
int release;
double icriteria;
double ii_distance;
double orth_fac;
double *mat_energy;
int cglog_level;
int interval;
int corrections;
static inline __attribute__((always_inline)) int cgas_malloc(size_t n, int s, int dupl, double *work, unsigned long works, double ***p, double ***tmp, double ***r, double ***alpha1, double ***alpha2) 
{
unsigned long ns = n * s;
unsigned long nsalgn = ((ns + 0x7f ) >> 7 ) << 7;
unsigned long salgn = ((s + 0x7f ) >> 7 ) << 7;
double **ptmp = malloc(5 * dupl * sizeof(double*));
if ( ptmp == NULL ) {
return 1;
}
int offs = 0;
double **lp = *p = &ptmp[offs];
offs += dupl;
double **lr = *r = &ptmp[offs];
offs += dupl;
double **ltmp = *tmp = &ptmp[offs];
offs += dupl;
double **lalpha1 = *alpha1 = &ptmp[offs];
offs += dupl;
double **lalpha2 = *alpha2 = &ptmp[offs];
unsigned long allocsize = 0;
int d;
for ( d=0; d<dupl; ++d ) {
lp[d] = work;
work += nsalgn;
allocsize += nsalgn; 
}
for ( d=0; d<dupl; ++d ) {
lr[d] = work;
work += nsalgn;
allocsize += nsalgn; 
}
for ( d=0; d<dupl; ++d ) {
ltmp[d] = work;
work += nsalgn;
allocsize += nsalgn; 
}
for ( d=0; d<dupl; ++d ) {
lalpha1[d] = work;
work += salgn;
lalpha2[d] = work;
work += salgn;
allocsize += salgn<<1; 
}
if ( allocsize > works ) {
fprintf(stderr, "err: insufficient workspace (avail %i req %i)\n", works, allocsize);
return 1;
}
return 0;
}
#define PRIOR_FIRSTCP		9999999
#define PRIOR_GEMM			99999
#define PRIOR_ASYNCDOT		999999
#define PRIOR_ASYNCAXPY		9999
typedef enum {
DOT_ALPHA1, 
DOT_ALPHA2,
DOT_ALPHA3, 
} dot_t;
typedef enum {
MAXITER,
CONVERGED,
IERROR,
} cg_exit_status;
double cg_ierror(int bm, int bn, int n, void *A, int s, void *b, void *xbuff, int *offs, double tol, int steps, void *work, unsigned long works, 
int async, double profile, int warmup, int cglog, int release_id, double icriteria, double distance, double orth_fac, double *mat_energy, 
int is_precond, void *zbuff, cs *Acs, css **S, csn **N, int interval, int corrections, int *bm_p) 
{
typedef struct { 
async_t sync[3];
double buff;
} worklay_t;
double *e_perc = malloc(steps*sizeof(double));
double *residuals = calloc(steps, sizeof(double));
double *ierror = calloc(steps, sizeof(double));
double *approx_res = calloc(n, sizeof(double));
double norm_b = cblas_ddot(n, b, 1, b, 1);
norm_b = sqrt(norm_b);
double min_ierror = DBL_MAX;
int cg_exit = MAXITER;
#if USE_SPARSE
hbmat_t *Ahb = A;
double *z[4] = {zbuff, zbuff+n, zbuff+2*n, zbuff+3*n};
#endif
double *lxbuff = xbuff;
double *x[4] = {lxbuff, lxbuff+n, lxbuff+2*n, lxbuff+3*n};
double **r; 
double **tmp;  
double **p; 
double **alpha1;
double **alpha2;
worklay_t *worklay = (worklay_t*) work;
double orth; 
double porth = DBL_MAX;
int dupl = 4;
if ( cgas_malloc(n, s, dupl, &worklay->buff, works, &p, &tmp, &r, &alpha1, &alpha2) ) {
return -1;
}
async_t *sync =  async_init(worklay->sync, 3, n, bm, -2, 1);
async_profile(&sync[DOT_ALPHA2], profile);
async_profile(&sync[DOT_ALPHA3], profile);
asman_t *asman = asman_init(4);
int i = asman_nexti(asman);
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, b, r[i]);
#if USE_SPARSE
bsblas_dcsrmv(PRIOR_GEMM, bm, FP_MONE, Ahb, x[i], FP_ONE, r[i]);
#else
bblas_dgemm(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[i], n, FP_ONE, r[i], n); 					
#endif
#if USE_SPARSE
if ( is_precond ) {
bsblas_dcholsolv2(1, bm, n, S, N, r[i], z[i]);
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, z[i], p[i]);
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
} else {
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]);
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
}
#else
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]); 				
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);            	
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
if ( ncorrection == corrections ) {
ncorrection = -1;
} else if ( ncorrection > -1 ) {
ncorrection += 1;
}
if ( k % interval == 0) {
ncorrection = 0;
}
}
i = i & 0x1;
if ( async && warmup<=0 && ncorrection<0) {
#if USE_SPARSE
bsblas_dcsrmv_prior_switch(&sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, mat_energy, bm, FP_ONE, Ahb, \
p[iprev], FP_NOUGHT, tmp[i], release_id, bm_p);
#else
bblas_dgemm_prior_switch(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, \
A, p[iprev], FP_NOUGHT, tmp[i], release_id, mat_energy, bm_p);
#endif
bblas_ddot(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 
bblas_scal_dcpaxpy_comb_async_concurrent(k, &sync[DOT_ALPHA2], PRIOR_ASYNCAXPY, bm, bn, n, s, \
FP_MONE, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
} else {
#if USE_SPARSE
bsblas_dcsrmv_prior(&sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, mat_energy, bm, FP_ONE, Ahb, p[iprev], FP_NOUGHT, tmp[i]);
#else
bblas_dgemm_prior(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], mat_energy);
#endif
bblas_ddot(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]);	
bblas_dcpaxpy_comb(bm, bn, n, s, FP_MONE, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
}
#if USE_SPARSE
if ( is_precond ) {
bsblas_dcholsolv2(1, bm, n, S, N, r[i], z[i]);
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], z[i], p[i]); 	
} else {
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
}
#else
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);             					
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
#endif
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, b, approx_res);
#if USE_SPARSE
bsblas_dcsrmv(PRIOR_GEMM, bm, FP_MONE, Ahb, x[i], FP_ONE, approx_res);
#else
bblas_dgemm(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[i], n, FP_ONE, approx_res, n);
#endif
#pragma omp taskwait
double norm_r = sqrt(*(alpha1[i]));
double sr2norm = norm_r/norm_b;
residuals[k] = sr2norm;
if (islessequal(sr2norm, tol)) {
cg_exit = CONVERGED;
is_converged = 1;
break;
}
int bs = (n+bm-1)/bm;
double acc_energy = (double)0;
double tol_energy = (double)0;
int tol_bm = 0;
for ( int i = 0; i < bs; i++ ) {
tol_bm += bm_p[i];
if ( bm_p[i] ) {
acc_energy += mat_energy[i];
} else {
bm_p[i] = 1;
}
tol_energy += mat_energy[i];
}
double energy_p = acc_energy/tol_energy;
e_perc[k] = energy_p;
double norm_approx_res = cblas_ddot(n, approx_res, 1, approx_res, 1);
norm_approx_res = sqrt(norm_approx_res);
ierror[k] = norm_approx_res;
if ( isless(norm_approx_res, min_ierror) ) {
min_ierror = norm_approx_res;
}
if ( k > 0 ) {
double lpi = log10(ierror[k-1]);
double cpi = log10(ierror[k]);
if ( isless(fabs(lpi - cpi), icriteria) ) {
cg_exit = IERROR;
break;
}
}
sync[DOT_ALPHA2].pcnt = sync[DOT_ALPHA3].pcnt = 0;
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
if ( warmup > 0 ) {
warmup--;
}
}
#pragma omp taskwait
free(p);
if ( cglog > 0 ) {
FILE *logr = fopen("cgprof_ierror.log", "a");
fprintf(logr, "iter ||A'x'-b|| ||Ax'-b|| ||b||\n");
for (int kk = 0; kk < k; kk++ ) {
fprintf(logr, "%d %E %E %E\n", kk, residuals[kk], ierror[kk], norm_b);
}
if ( cg_exit == IERROR ) {
fprintf(logr, "Ierror stable %E\n", icriteria);
}
if (isgreater(min_ierror/norm_b, distance)) {
fprintf(logr, "too close set to ||b||\n");
min_ierror = norm_b;
}
fprintf(logr, "min %E %E\n", min_ierror, norm_b);
fprintf(logr, "ITREF iteration ends\n");
fclose(logr);
}
free(residuals);
free(ierror);
free(e_perc);
return min_ierror;
}
int dcg_itref(int bm, int bn, int n, void *A, int s, void *b, void *xbuff, int *offs, double tol, int steps, void *work, unsigned long works, 
int async, double profile, int warmup, int cglog, int release, double orth_fac, double *mat_energy, 
int is_precond, void *zbuff, cs *Acs, css **S, csn **N, int interval, int corrections, int *bm_p, double acrit) 
{
typedef struct { 
async_t sync[3];
double buff;
} worklay_t;
double *e_perc = malloc(steps*sizeof(double));
double *residuals = calloc(steps, sizeof(double));
double *ierror = calloc(steps, sizeof(double));
double *approx_res = calloc(n, sizeof(double));
double norm_b = cblas_ddot(n, b, 1, b, 1);
norm_b = sqrt(norm_b);
double min_ierror = DBL_MAX;
double icriteria = 1E-5;
int *approx_log = calloc(steps, sizeof(int));
int cg_exit = MAXITER;
#if USE_SPARSE
hbmat_t *Ahb = A;
double *z[4] = {zbuff, zbuff+n, zbuff+2*n, zbuff+3*n};
#endif
double *lxbuff = xbuff;
double *x[4] = {lxbuff, lxbuff+n, lxbuff+2*n, lxbuff+3*n};
double **r; 
double **tmp;  
double **p; 
double **alpha1;
double **alpha2;
worklay_t *worklay = (worklay_t*) work;
double orth; 
double porth = DBL_MAX;
int dupl = 4;
if ( cgas_malloc(n, s, dupl, &worklay->buff, works, &p, &tmp, &r, &alpha1, &alpha2) ) {
return -1;
}
async_t *sync =  async_init(worklay->sync, 3, n, bm, -2, 1);
async_profile(&sync[DOT_ALPHA2], profile);
async_profile(&sync[DOT_ALPHA3], profile);
asman_t *asman = asman_init(4);
int i = asman_nexti(asman);
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, b, r[i]);
#if USE_SPARSE
bsblas_dcsrmv(PRIOR_GEMM, bm, FP_MONE, Ahb, x[i], FP_ONE, r[i]);
#else
bblas_dgemm(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[i], n, FP_ONE, r[i], n); 					
#endif
#if USE_SPARSE
if ( is_precond ) {
bsblas_dcholsolv2(1, bm, n, S, N, r[i], z[i]);
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, z[i], p[i]);
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
} else {
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]);
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
}
#else
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]); 				
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);            	
#endif
int iprev = 0;
int aheadc = 0;
async_stat_t status = STAT_SYNC;
asman_update(asman, 0, 0);
int is_converged = 0;
int ncorrection = -1;
int aswitch = 0;
int k; 
for ( k=0; k<steps; ++k ) {
i = i & 0x1;
if ( async && warmup <= 0 ) {
if ( k == 0 ) {
aswitch = isless(acrit, norm_b);
} else {
aswitch = isless(acrit, residuals[k-1]);
}
}
if ( aswitch ) {
#if USE_SPARSE
bsblas_dcsrmv_prior_switch(&sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, mat_energy, bm, FP_ONE, Ahb, p[iprev], FP_NOUGHT, tmp[i], release, bm_p);
#else
bblas_dgemm_prior_switch(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], release, mat_energy, bm_p);
#endif
bblas_ddot(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
bblas_scal_dcpaxpy_comb_async_concurrent(k, &sync[DOT_ALPHA2], PRIOR_ASYNCAXPY, bm, bn, n, s, FP_MONE, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
} else {
#if USE_SPARSE
bsblas_dcsrmv_prior(&sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, mat_energy, bm, FP_ONE, Ahb, p[iprev], FP_NOUGHT, tmp[i]);
#else
bblas_dgemm_prior(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], mat_energy);
#endif
bblas_ddot(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
bblas_dcpaxpy_comb(bm, bn, n, s, FP_MONE, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
}
#if USE_SPARSE
if ( is_precond ) {
bsblas_dcholsolv2(1, bm, n, S, N, r[i], z[i]);
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], z[i], p[i]); 	
} else {
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
}
#else
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);             					
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
#endif
#pragma omp taskwait
BLAS_gemm(OMPSSBLAS_TRANSP, OMPSSBLAS_NTRANSP, s, s, n, FP_ONE, p[i], n, tmp[i], n, FP_NOUGHT, &orth, s);
orth = FP_ABS(orth);
if ( cglog > 1 ) {
fprintf(stdout, "orth: %e\n", orth);
}
if( warmup <= 0 ) {
if (isgreater(orth, porth * orth_fac)){
fprintf(stdout, "orth fail\n", orth);
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
porth = orth;
double norm_r = sqrt(*(alpha1[i]));
double sr2norm = norm_r/norm_b;
residuals[k] = sr2norm;
if (islessequal(sr2norm, tol)) {
cg_exit = CONVERGED;
is_converged = 1;
break;
}
int bs = (n+bm-1)/bm;
double acc_energy = (double)0;
double tol_energy = (double)0;
int tol_bm = 0;
for ( int i = 0; i < bs; i++ ) {
tol_bm += bm_p[i];
if ( bm_p[i] ) {
acc_energy += mat_energy[i];
} else {
bm_p[i] = 1;
}
tol_energy += mat_energy[i];
}
double energy_p = acc_energy/tol_energy;
e_perc[k] = energy_p;
approx_log[k] = aswitch;
sync[DOT_ALPHA2].pcnt = sync[DOT_ALPHA3].pcnt = 0;
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
if ( warmup > 0 ) {
warmup--;
}
}
#pragma omp taskwait
free(p);
if ( cglog > 0 ) {
int approx = 0;
int compl = 0;
FILE *logr = fopen("cgprof_residual.log", "a");
for (int kk = 0; kk < k; kk++ ) {
fprintf(logr, "%d %E %d\n", kk, residuals[kk], approx_log[kk]);
approx_log[kk] ? approx++ : compl++;
}
fprintf(logr, "%d %d %.4f\n", approx, compl, (double)approx/(double)(approx+compl));
fprintf(logr, "ITREF iteration ends\n");
fclose(logr);
}
if ( cglog > 0 ) {
FILE *logr = fopen("cgprof_energy.log", "a");
for (int kk = 0; kk < k; kk++ ) {
fprintf(logr, "%d: %E\n", kk, e_perc[kk]);
}
fprintf(logr, "ITREF iteration ends\n");
fclose(logr);
}
free(residuals);
free(ierror);
free(e_perc);
k = is_converged ? -1 : k;
return k;
}
int dcg_pure(int bm, int bn, int n, void *A, int s, void *b, void *xbuff, int *offs, double tol, int steps, void *work, unsigned long works, 
int async, double profile, int warmup, int cglog, int release, double orth_fac, double *mat_energy, 
int is_precond, void *zbuff, cs *Acs, css **S, csn **N, int *bm_p) 
{
typedef struct { 
async_t sync[3];
double buff;
} worklay_t;
double *e_perc = malloc(steps*sizeof(double));
double *residuals = calloc(steps, sizeof(double));
double *ierror = calloc(steps, sizeof(double));
double *approx_res = calloc(n, sizeof(double));
double norm_b = cblas_ddot(n, b, 1, b, 1);
norm_b = sqrt(norm_b);
int *approx_log = calloc(steps, sizeof(int));
int cg_exit = MAXITER;
struct timeval stop, start;
unsigned long *cglaps = calloc(steps, sizeof(unsigned long));
#if USE_SPARSE
hbmat_t *Ahb = A;
double *z[4] = {zbuff, zbuff+n, zbuff+2*n, zbuff+3*n};
#endif
double *lxbuff = xbuff;
double *x[4] = {lxbuff, lxbuff+n, lxbuff+2*n, lxbuff+3*n};
double **r; 
double **tmp;  
double **p; 
double **alpha1;
double **alpha2;
worklay_t *worklay = (worklay_t*) work;
double orth; 
double porth = DBL_MAX;
int dupl = 4;
if ( cgas_malloc(n, s, dupl, &worklay->buff, works, &p, &tmp, &r, &alpha1, &alpha2) ) {
return -1;
}
async_t *sync =  async_init(worklay->sync, 3, n, bm, -2, 1);
async_profile(&sync[DOT_ALPHA2], profile);
async_profile(&sync[DOT_ALPHA3], profile);
asman_t *asman = asman_init(4);
int i = asman_nexti(asman);
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, b, r[i]);
#if USE_SPARSE
bsblas_dcsrmv(PRIOR_GEMM, bm, FP_MONE, Ahb, x[i], FP_ONE, r[i]);
#else
bblas_dgemm(PRIOR_FIRSTCP, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[i], n, FP_ONE, r[i], n); 					
#endif
#if USE_SPARSE
if ( is_precond ) {
bsblas_dcholsolv2(1, bm, n, S, N, r[i], z[i]);
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, z[i], p[i]);
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
} else {
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]);
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
}
#else
bblas_dcopy(PRIOR_FIRSTCP, bm, bn, n, s, r[i], p[i]); 				
bblas_ddot(-1, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);            	
#endif
int iprev = 0;
int aheadc = 0;
async_stat_t status = STAT_SYNC;
asman_update(asman, 0, 0);
int is_converged = 0;
int ncorrection = -1;
int aswitch = 0;
int k; 
for ( k=0; k<steps; ++k ) {
gettimeofday(&start, NULL);
i = i & 0x1;
if (async && warmup <= 0) {
#if USE_SPARSE
bsblas_dcg_comb_switch(k, &sync[DOT_ALPHA2], bm, FP_ONE, Ahb, p[iprev], FP_NOUGHT, tmp[i], alpha2[i], release);
#else
bblas_dgemm_prior_switch(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], release, mat_energy, bm_p);
bblas_ddot(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
#endif
bblas_scal_dcpaxpy_comb_async_concurrent(k, &sync[DOT_ALPHA2], PRIOR_ASYNCAXPY, bm, bn, n, s, FP_MONE, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
} else {
#if USE_SPARSE
bsblas_dcg_comb(k, &sync[DOT_ALPHA2], bm, FP_ONE, Ahb, p[iprev], FP_NOUGHT, tmp[i], alpha2[i]);
#else
bblas_dgemm_prior(k, &sync[DOT_ALPHA2], &sync[DOT_ALPHA3], PRIOR_GEMM, bm, bn, bm, n, s, n, FP_ONE, A, p[iprev], FP_NOUGHT, tmp[i], mat_energy);
bblas_ddot(k, &sync[DOT_ALPHA2], bm, bn, n, s, tmp[i], p[iprev], alpha2[i]); 		
#endif
bblas_dcpaxpy_comb(bm, bn, n, s, FP_MONE, alpha1[iprev], alpha2[i], tmp[i], p[iprev], r[iprev], x[iprev], r[i], x[i]);    	
}
#if USE_SPARSE
if ( is_precond ) {
bsblas_dcholsolv2(1, bm, n, S, N, r[i], z[i]);
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], z[i], alpha1[i]);
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], z[i], p[i]); 	
} else {
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
}
#else
bblas_ddot(k, &sync[DOT_ALPHA1], bm, bn, n, s, r[i], r[i], alpha1[i]);             					
bblas_extm_daxpy(PRIOR_ASYNCDOT, bm, bn, n, s, alpha1[i], alpha1[iprev], p[iprev], r[i], p[i]); 	
#endif
#pragma omp taskwait
gettimeofday(&stop, NULL);
cglaps[k] = (stop.tv_sec - start.tv_sec) * 1e6 + stop.tv_usec - start.tv_usec;
porth = orth;
double norm_r = sqrt(*(alpha1[i]));
double sr2norm = norm_r/norm_b;
residuals[k] = sr2norm;
if (islessequal(sr2norm, tol)) {
cg_exit = CONVERGED;
is_converged = 1;
break;
}
int bs = (n+bm-1)/bm;
double acc_energy = (double)0;
double tol_energy = (double)0;
int tol_bm = 0;
for ( int i = 0; i < bs; i++ ) {
tol_bm += bm_p[i];
if ( bm_p[i] ) {
acc_energy += mat_energy[i];
} else {
bm_p[i] = 1;
}
tol_energy += mat_energy[i];
}
double energy_p = acc_energy/tol_energy;
e_perc[k] = energy_p;
approx_log[k] = aswitch;
sync[DOT_ALPHA2].pcnt = sync[DOT_ALPHA3].pcnt = 0;
iprev = i;
aheadc += status == STAT_AHEAD ? 1 : 0;
if ( warmup > 0 ) {
warmup--;
}
}
#pragma omp taskwait
free(p);
if ( cglog > 0 ) {
int approx = 0;
int compl = 0;
FILE *logr = fopen("cgprof_residual.log", "a");
for (int kk = 0; kk < k; kk++ ) {
fprintf(logr, "%d %E %d\n", kk, residuals[kk], cglaps[kk]);
approx_log[kk] ? approx++ : compl++;
}
fprintf(logr, "%d %d %.4f\n", approx, compl, (double)approx/(double)(approx+compl));
fprintf(logr, "ITREF iteration ends\n");
fclose(logr);
}
if ( cglog > 0 ) {
FILE *logr = fopen("cgprof_energy.log", "a");
for (int kk = 0; kk < k; kk++ ) {
fprintf(logr, "%d: %E\n", kk, e_perc[kk]);
}
fprintf(logr, "ITREF iteration ends\n");
fclose(logr);
}
free(residuals);
free(ierror);
free(e_perc);
k = is_converged ? -1 : k;
return k;
}
int main (int argc, char *argv[]) 
{
if ( itref_config(argc, argv) ) {
return 1;
}
bn = s = 1;
if ( itref_dsetup(n, bm, bn, s, &A, x, &rhs, &work) ) {
return 2;
}
int n2 = n * n;
int ns = n * s;
struct timeval start;
double *sdb = malloc(2 * ns * 4 * sizeof(double));
double *sd[] = { &sdb[0], &sdb[4*ns] };
double *rb = malloc(2 * ns * sizeof(double));
double *r[] = { &rb[0], &rb[ns] };
double *rpb = malloc(2*ns*sizeof(double));
double *rp[] = {&rpb[0], &rpb[ns]};
double *srb = malloc(2 * ns * sizeof(double));
double *sr[] = { &srb[0], &srb[ns] };
double *r2norm = calloc(2, sizeof(double));
double *r2normp = calloc(2, sizeof(double));
double *norms = malloc(refit * sizeof(double));
double *normps = malloc(refit * sizeof(double));
double *sz = malloc(ns * 4 * sizeof(double));
int *bm_p = calloc((n+bm-1)/bm, sizeof(int));
double acrit = (double)0;
double *sdb2 = malloc(2 * ns * 4 * sizeof(double));
double *sd2[] = { &sdb2[0], &sdb2[4*ns] };
async_t *sync = malloc(3*sizeof(async_t));
async_setup(sync);
async_init(sync, 3, n, bm, -1, 1);
async_log(&sync[0], 256, "itref");
unsigned long elapsed=0;
int offs = 0;
int is_converged = 0;
int k;
for ( k=0; k<rep; ++k ) {
memset(sd[1], 0, ns * 4 * sizeof(double));
memset(sd[0], 0, ns * 4 * sizeof(double));
if ( refit == 0 ) {
int bs = (n+bm-1)/bm;
for ( int release_id = 0; release_id < bs; release_id++ ) {
#if USE_SPARSE
dcg_pure(bm, bn, n, Acsr, s, rhs, sd[1], &offs, prec, it, work, works, async, profile, warmup, cglog_level, \
release_id, orth_fac, mat_energy, is_precond, sz, Acs, S, N, bm_p);
#else
dcg_pure(bm, bn, n, A, s, rhs, sd[1], &offs, prec, it, work, works, async, profile, warmup, cglog_level, \
release_id, orth_fac, mat_energy, 0, NULL, NULL, NULL, NULL, bm_p);
#endif
}
}
if ( refit > 0 ) {
#if USE_SPARSE
dcg_pure(bm, bn, n, Acsr, s, rhs, sd[1], &offs, prec, it, work, works, async, profile, warmup, cglog_level, \
release, orth_fac, mat_energy, is_precond, sz, Acs, S, N, bm_p);
#else
dcg_pure(bm, bn, n, A, s, rhs, sd[1], &offs, prec, it, work, works, async, profile, warmup, cglog_level, \
release, orth_fac, mat_energy, 0, NULL, NULL, NULL, NULL, bm_p);
#endif
bblas_dcopy(1, bm, bn, n, s, sd[1], x[1]);
bblas_dcopy(1, bm, bn, n, s, rhs, r[1]);
bblas_dcopy(1, bm, bn, n, s, rhs, rp[1]);
#if USE_SPARSE
bsblas_dcsrmv(1, bm, FP_MONE, Acsr, x[1], FP_ONE, r[1]);
bsblas_dcsrmv_switch(bm, FP_MONE, Acsr, x[1], FP_ONE, rp[1], release);
#else
bblas_dgemm(1, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[1], n, FP_ONE, r[1], n); 	
#endif
int broken = 0;  
int brokec = 0;
int brokeseq = 0;
int prevb = 1;
r2norm[1] = DBL_MAX; 
r2normp[1] = DBL_MAX;
int i;
for ( i=0; i<refit; ++i ) {
fprintf(stdout, "ITREF i: %d\n", i);
int b = i & 0x1;
gettimeofday(&start, NULL);
bblas_dcopy(1, bm, bn, n, s, r[prevb], sr[b]);
bblas_dset(bm, bn, n, 4 * s, 0.0, sd[b]);
int release_id = release;
if ( i > 0 && isless(normps[i-1], norms[i-1])) 
release_id = -1;
#if USE_SPARSE
int cgit = dcg_pure(bm, bn, n, Acsr, s, sr[b], sd[b], &offs, prec, it, work, works, async, profile, warmup, \
cglog_level, release_id, orth_fac, mat_energy, is_precond, sz, Acs, S, N, bm_p);
#else
int cgit = dcg_pure(bm, bn, n, A, s, sr[b], sd[b], &offs, prec, it, work, works, async, profile, warmup, \
cglog_level, release_id, orth_fac, mat_energy, 0, NULL, NULL, NULL, NULL, bm_p);
#endif
#pragma omp taskwait on (r2norm[prevb], r2normp[prevb])
double sr2norm = sqrt(r2norm[prevb]);
double sr2normp = sqrt(r2normp[prevb]);
norms[i] = sr2norm; normps[i] = sr2normp;
struct timeval stop;
gettimeofday(&stop, NULL);
unsigned long itlaps = (stop.tv_sec - start.tv_sec) * 1e6 + stop.tv_usec - start.tv_usec;
log_record(&sync[0], i-1, EVENT_RESIDUAL, itlaps, sr2norm);
if ( islessequal(sr2norm, prec) ) {
break;
}
bblas_sdcpaxpy(bm, bn, n, s, FP_ONE, sd[b], x[prevb], x[b]); 
bblas_dcopy(1, bm, bn, n, s, rhs, r[b]);
bblas_dcopy(1, bm, bn, n, s, rhs, rp[b]);
#if USE_SPARSE
bsblas_dcsrmv(1, bm, FP_MONE, Acsr, x[b], FP_ONE, r[b]);
bsblas_dcsrmv_switch(bm, FP_MONE, Acsr, x[b], FP_ONE, rp[b], release);
#else
bblas_dgemm(1, OMPSSBLAS_NTRANSP, OMPSSBLAS_NTRANSP, bm, bn, bm, n, s, n, FP_MONE, A, n, x[b], n, FP_ONE, r[b], n); 	
#endif
bblas_ddot(i, &sync[1], bm, bn, n, s, r[b], r[b], &r2norm[b]);
bblas_ddot(i, &sync[2], bm, bn, n, s, rp[b], rp[b], &r2normp[b]);
prevb = b;
}
}
#pragma omp taskwait
}
FILE *logr = fopen("itref_norms.log", "w");
fprintf(logr, "iter: ||Ax-b|| ||A'x-b||\n");
for (int kk = 0; kk < refit; kk++ ) {
fprintf(logr, "%d: %E %E\n", kk, norms[kk], normps[kk]);
}
fclose(logr);
free(mat_energy);
free(sdb);
free(rb);
free(sz);
free(sdb2);
async_fini(sync, 3);
#if USE_SPARSE
itref_cleanup(n, s, Acsr, x, rhs, work);
#else
itref_cleanup(n, s, A, x, rhs, work);
#endif
return 0;
}
