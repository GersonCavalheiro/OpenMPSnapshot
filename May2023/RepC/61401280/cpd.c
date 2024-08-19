#include "base.h"
#include "cpd.h"
#include "matrix.h"
#include "mttkrp.h"
#include "timer.h"
#include "thd_info.h"
#include "util.h"
#include <math.h>
int splatt_cpd_als(
splatt_csf const * const tensors,
splatt_idx_t const nfactors,
double const * const options,
splatt_kruskal * factored)
{
matrix_t * mats[MAX_NMODES+1];
idx_t nmodes = tensors->nmodes;
rank_info rinfo;
rinfo.rank = 0;
idx_t maxdim = tensors->dims[argmax_elem(tensors->dims, nmodes)];
for(idx_t m=0; m < nmodes; ++m) {
mats[m] = (matrix_t *) mat_rand(tensors[0].dims[m], nfactors);
}
mats[MAX_NMODES] = mat_alloc(maxdim, nfactors);
val_t * lambda = (val_t *) splatt_malloc(nfactors * sizeof(val_t));
factored->fit = cpd_als_iterate(tensors, mats, lambda, nfactors, &rinfo,
options);
factored->rank = nfactors;
factored->nmodes = nmodes;
factored->lambda = lambda;
for(idx_t m=0; m < nmodes; ++m) {
factored->dims[m] = tensors->dims[m];
factored->factors[m] = mats[m]->vals;
}
mat_free(mats[MAX_NMODES]);
for(idx_t m=0; m < nmodes; ++m) {
free(mats[m]); 
}
return SPLATT_SUCCESS;
}
void splatt_free_kruskal(
splatt_kruskal * factored)
{
free(factored->lambda);
for(idx_t m=0; m < factored->nmodes; ++m) {
free(factored->factors[m]);
}
}
static void p_reset_cpd_timers(
rank_info const * const rinfo)
{
timer_reset(&timers[TIMER_ATA]);
#ifdef SPLATT_USE_MPI
timer_reset(&timers[TIMER_MPI]);
timer_reset(&timers[TIMER_MPI_IDLE]);
timer_reset(&timers[TIMER_MPI_COMM]);
timer_reset(&timers[TIMER_MPI_ATA]);
timer_reset(&timers[TIMER_MPI_REDUCE]);
timer_reset(&timers[TIMER_MPI_NORM]);
timer_reset(&timers[TIMER_MPI_UPDATE]);
timer_reset(&timers[TIMER_MPI_FIT]);
MPI_Barrier(rinfo->comm_3d);
#endif
}
static val_t p_kruskal_norm(
idx_t const nmodes,
val_t const * const restrict lambda,
matrix_t ** aTa)
{
idx_t const rank = aTa[0]->J;
val_t * const restrict av = aTa[MAX_NMODES]->vals;
val_t norm_mats = 0;
for(idx_t i=0; i < rank; ++i) {
for(idx_t j=i; j < rank; ++j) {
av[j + (i*rank)] = 1.;
}
}
for(idx_t m=0; m < nmodes; ++m) {
val_t const * const restrict atavals = aTa[m]->vals;
for(idx_t i=0; i < rank; ++i) {
for(idx_t j=i; j < rank; ++j) {
av[j + (i*rank)] *= atavals[j + (i*rank)];
}
}
}
for(idx_t i=0; i < rank; ++i) {
norm_mats += av[i+(i*rank)] * lambda[i] * lambda[i];
for(idx_t j=i+1; j < rank; ++j) {
norm_mats += av[j+(i*rank)] * lambda[i] * lambda[j] * 2;
}
}
return fabs(norm_mats);
}
static val_t p_tt_kruskal_inner(
idx_t const nmodes,
rank_info * const rinfo,
thd_info * const thds,
val_t const * const restrict lambda,
matrix_t ** mats,
matrix_t const * const m1)
{
idx_t const rank = mats[0]->J;
idx_t const lastm = nmodes - 1;
idx_t const dim = m1->I;
val_t const * const m0 = mats[lastm]->vals;
val_t const * const mv = m1->vals;
val_t myinner = 0;
#pragma omp parallel reduction(+:myinner)
{
int const tid = splatt_omp_get_thread_num();
val_t * const restrict accumF = (val_t *) thds[tid].scratch[0];
for(idx_t r=0; r < rank; ++r) {
accumF[r] = 0.;
}
#pragma omp for
for(idx_t i=0; i < dim; ++i) {
for(idx_t r=0; r < rank; ++r) {
accumF[r] += m0[r+(i*rank)] * mv[r+(i*rank)];
}
}
for(idx_t r=0; r < rank; ++r) {
myinner += accumF[r] * lambda[r];
}
}
val_t inner = 0.;
#ifdef SPLATT_USE_MPI
timer_start(&timers[TIMER_MPI_FIT]);
MPI_Allreduce(&myinner, &inner, 1, SPLATT_MPI_VAL, MPI_SUM, rinfo->comm_3d);
timer_stop(&timers[TIMER_MPI_FIT]);
#else
inner = myinner;
#endif
return inner;
}
static val_t p_calc_fit(
idx_t const nmodes,
rank_info * const rinfo,
thd_info * const thds,
val_t const ttnormsq,
val_t const * const restrict lambda,
matrix_t ** mats,
matrix_t const * const m1,
matrix_t ** aTa)
{
timer_start(&timers[TIMER_FIT]);
val_t const norm_mats = p_kruskal_norm(nmodes, lambda, aTa);
val_t const inner = p_tt_kruskal_inner(nmodes, rinfo, thds, lambda, mats,m1);
val_t residual = ttnormsq + norm_mats - (2 * inner);
if(residual > 0.) {
residual = sqrt(residual);
}
timer_stop(&timers[TIMER_FIT]);
return 1 - (residual / sqrt(ttnormsq));
}
double cpd_als_iterate(
splatt_csf const * const tensors,
matrix_t ** mats,
val_t * const lambda,
idx_t const nfactors,
rank_info * const rinfo,
double const * const opts)
{
idx_t const nmodes = tensors[0].nmodes;
idx_t const nthreads = (idx_t) opts[SPLATT_OPTION_NTHREADS];
splatt_omp_set_num_threads(nthreads);
thd_info * thds =  thd_init(nthreads, 3,
(nmodes * nfactors * sizeof(val_t)) + 64,
0,
(nmodes * nfactors * sizeof(val_t)) + 64);
matrix_t * m1 = mats[MAX_NMODES];
matrix_t * aTa[MAX_NMODES+1];
for(idx_t m=0; m < nmodes; ++m) {
aTa[m] = mat_alloc(nfactors, nfactors);
memset(aTa[m]->vals, 0, nfactors * nfactors * sizeof(val_t));
mat_aTa(mats[m], aTa[m], rinfo, thds, nthreads);
}
aTa[MAX_NMODES] = mat_alloc(nfactors, nfactors);
splatt_mttkrp_ws * mttkrp_ws = splatt_mttkrp_alloc_ws(tensors,nfactors,opts);
double oldfit = 0;
double fit = 0;
val_t ttnormsq = csf_frobsq(tensors);
p_reset_cpd_timers(rinfo);
sp_timer_t itertime;
sp_timer_t modetime[MAX_NMODES];
timer_start(&timers[TIMER_CPD]);
idx_t const niters = (idx_t) opts[SPLATT_OPTION_NITER];
for(idx_t it=0; it < niters; ++it) {
timer_fstart(&itertime);
for(idx_t m=0; m < nmodes; ++m) {
timer_fstart(&modetime[m]);
mats[MAX_NMODES]->I = tensors[0].dims[m];
m1->I = mats[m]->I;
timer_start(&timers[TIMER_MTTKRP]);
mttkrp_csf(tensors, mats, m, thds, mttkrp_ws, opts);
timer_stop(&timers[TIMER_MTTKRP]);
#if 0
calc_gram_inv(m, nmodes, aTa);
memset(mats[m]->vals, 0, mats[m]->I * nfactors * sizeof(val_t));
mat_matmul(m1, aTa[MAX_NMODES], mats[m]);
#else
par_memcpy(mats[m]->vals, m1->vals, m1->I * nfactors * sizeof(val_t));
mat_solve_normals(m, nmodes, aTa, mats[m],
opts[SPLATT_OPTION_REGULARIZE]);
#endif
if(it == 0) {
mat_normalize(mats[m], lambda, MAT_NORM_2, rinfo, thds, nthreads);
} else {
mat_normalize(mats[m], lambda, MAT_NORM_MAX, rinfo, thds,nthreads);
}
mat_aTa(mats[m], aTa[m], rinfo, thds, nthreads);
timer_stop(&modetime[m]);
} 
fit = p_calc_fit(nmodes, rinfo, thds, ttnormsq, lambda, mats, m1, aTa);
timer_stop(&itertime);
if(rinfo->rank == 0 &&
opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_NONE) {
printf("  its = %3"SPLATT_PF_IDX" (%0.3fs)  fit = %0.5f  delta = %+0.4e\n",
it+1, itertime.seconds, fit, fit - oldfit);
if(opts[SPLATT_OPTION_VERBOSITY] > SPLATT_VERBOSITY_LOW) {
for(idx_t m=0; m < nmodes; ++m) {
printf("     mode = %1"SPLATT_PF_IDX" (%0.3fs)\n", m+1,
modetime[m].seconds);
}
}
}
if(fit == 1. || 
(it > 0 && fabs(fit - oldfit) < opts[SPLATT_OPTION_TOLERANCE])) {
break;
}
oldfit = fit;
}
timer_stop(&timers[TIMER_CPD]);
cpd_post_process(nfactors, nmodes, mats, lambda, thds, nthreads, rinfo);
splatt_mttkrp_free_ws(mttkrp_ws);
for(idx_t m=0; m < nmodes; ++m) {
mat_free(aTa[m]);
}
mat_free(aTa[MAX_NMODES]);
thd_free(thds, nthreads);
return fit;
}
void cpd_post_process(
idx_t const nfactors,
idx_t const nmodes,
matrix_t ** mats,
val_t * const lambda,
thd_info * const thds,
idx_t const nthreads,
rank_info * const rinfo)
{
val_t * tmp =  splatt_malloc(nfactors * sizeof(*tmp));
for(idx_t m=0; m < nmodes; ++m) {
mat_normalize(mats[m], tmp, MAT_NORM_2, rinfo, thds, nthreads);
for(idx_t f=0; f < nfactors; ++f) {
lambda[f] *= tmp[f];
}
}
free(tmp);
}
