#include "thd_info.h"
static inline void p_reduce_sum(
thd_info * const thds,
idx_t const scratchid,
idx_t const nelems)
{
int const tid = splatt_omp_get_thread_num();
int const nthreads = splatt_omp_get_num_threads();
val_t * const myvals = (val_t *) thds[tid].scratch[scratchid];
int half = nthreads / 2;
while(half > 0) {
if(tid < half && tid + half < nthreads) {
val_t const * const target = (val_t *) thds[tid+half].scratch[scratchid];
for(idx_t i=0; i < nelems; ++i) {
myvals[i] += target[i];
}
}
#pragma omp barrier
#pragma omp master
if(half > 1 && half % 2 == 1) {
val_t const * const last = (val_t *) thds[half-1].scratch[scratchid];
for(idx_t i=0; i < nelems; ++i) {
myvals[i] += last[i];
}
}
half /= 2;
}
#pragma omp master
{
if(nthreads % 2 == 1) {
val_t const * const last = (val_t *) thds[nthreads-1].scratch[scratchid];
for(idx_t i=0; i < nelems; ++i) {
myvals[i] += last[i];
}
}
}
#pragma omp barrier
}
static inline void p_reduce_max(
thd_info * const thds,
idx_t const scratchid,
idx_t const nelems)
{
int const tid = splatt_omp_get_thread_num();
int const nthreads = splatt_omp_get_num_threads();
val_t * const myvals = (val_t *) thds[tid].scratch[scratchid];
int half = nthreads / 2;
while(half > 0) {
if(tid < half && tid + half < nthreads) {
val_t const * const target = (val_t *) thds[tid+half].scratch[scratchid];
for(idx_t i=0; i < nelems; ++i) {
myvals[i] = SS_MAX(myvals[i], target[i]);
}
}
#pragma omp barrier
#pragma omp master
if(half > 1 && half % 2 == 1) {
val_t const * const last = (val_t *) thds[half-1].scratch[scratchid];
for(idx_t i=0; i < nelems; ++i) {
myvals[i] = SS_MAX(myvals[i], last[i]);
}
}
half /= 2;
}
#pragma omp master
{
if(nthreads % 2 == 1) {
val_t const * const last = (val_t *) thds[nthreads-1].scratch[scratchid];
for(idx_t i=0; i < nelems; ++i) {
myvals[i] = SS_MAX(myvals[i], last[i]);
}
}
}
#pragma omp barrier
}
void thd_reduce(
thd_info * const thds,
idx_t const scratchid,
idx_t const nelems,
splatt_reduce_type const which)
{
if(splatt_omp_get_num_threads() == 1) {
return;
}
#pragma omp barrier
switch(which) {
case REDUCE_SUM:
p_reduce_sum(thds, scratchid, nelems);
break;
case REDUCE_MAX:
p_reduce_max(thds, scratchid, nelems);
break;
default:
fprintf(stderr, "SPLATT: thd_reduce supports SUM and MAX only.\n");
abort();
}
}
thd_info * thd_init(
idx_t const nthreads,
idx_t const nscratch,
...)
{
thd_info * thds = (thd_info *) splatt_malloc(nthreads * sizeof(thd_info));
for(idx_t t=0; t < nthreads; ++t) {
timer_reset(&thds[t].ttime);
thds[t].nscratch = nscratch;
thds[t].scratch = (void **) splatt_malloc(nscratch * sizeof(void*));
}
va_list args;
va_start(args, nscratch);
for(idx_t s=0; s < nscratch; ++s) {
idx_t const bytes = va_arg(args, idx_t);
for(idx_t t=0; t < nthreads; ++t) {
thds[t].scratch[s] = (void *) splatt_malloc(bytes);
memset(thds[t].scratch[s], 0, bytes);
}
}
va_end(args);
return thds;
}
void thd_times(
thd_info * thds,
idx_t const nthreads)
{
for(idx_t t=0; t < nthreads; ++t) {
printf("  thread: %"SPLATT_PF_IDX" %0.3fs\n", t, thds[t].ttime.seconds);
}
}
void thd_time_stats(
thd_info * thds,
idx_t const nthreads)
{
double max_time = 0.;
double avg_time = 0.;
for(idx_t t=0; t < nthreads; ++t) {
avg_time += thds[t].ttime.seconds;
max_time = SS_MAX(max_time, thds[t].ttime.seconds);
}
avg_time /= nthreads;
double const imbal = (max_time - avg_time) / max_time;
printf("  avg: %0.3fs max: %0.3fs (%0.1f%% imbalance)\n",
avg_time, max_time, 100. * imbal);
}
void thd_reset(
thd_info * thds,
idx_t const nthreads)
{
for(idx_t t=0; t < nthreads; ++t) {
timer_reset(&thds[t].ttime);
}
}
void thd_free(
thd_info * thds,
idx_t const nthreads)
{
for(idx_t t=0; t < nthreads; ++t) {
for(idx_t s=0; s < thds[t].nscratch; ++s) {
free(thds[t].scratch[s]);
}
free(thds[t].scratch);
}
free(thds);
}
