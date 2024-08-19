#include "task_dot_async.h"
#include "fptype.h"
#include "fpblas.h"
#include "blas.h"
#include "task_log.h"
#include "async_struct.h"
#include <pthread.h>
#include <stdio.h>
#ifdef SINGLE_PRECISION 
#define __t_dot_async 			task_sdot_async
#define __t_dot_sched_async 	task_sdot_sched_async
#define __t_dot_sched_async_release 	task_sdot_sched_async_release
#define __t_dot_sched_async_concurrent	task_sdot_sched_async_concurrent
#define __t_dot3_async 			task_sdot3_async
#else
#define __t_dot_async 			task_ddot_async
#define __t_dot_sched_async 	task_ddot_sched_async
#define __t_dot_sched_async_release 	task_ddot_sched_async_release
#define __t_dot_sched_async_concurrent	task_ddot_sched_async_concurrent
#define __t_dot3_async 			task_ddot3_async
#endif
void __t_dot_async(int id, async_t *dotid, int p, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *result) 
{
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
int wait;
int log = 0;
int pcnt;
int ready = 0;
pthread_mutex_t *mutex = &dotid->mutex;
pthread_mutex_lock(mutex);
if ( dotid->consume != id ) {
if ( dotid->create == id ) {
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
} else {
BLAS_copy(bn, local_result, i_one, result, i_one);
dotid->create = id;
dotid->pcnt = 0;
}
if ( dotid->wait > 0 ) {
log = 1;
dotid->consume = id;
dotid->wait = 0;
pthread_cond_broadcast(&dotid->cond);
}
}
pcnt = ++dotid->pcnt;
ready = pcnt == dotid->pcompl;	
dotid->ready += ready? 1: 0;
pthread_mutex_unlock(mutex);
if ( log && dotid->log ) {
fp_t fpcnt = pcnt;
#if 0
int scale = dotid->flags & ASYNC_SCALE;
fp_t factor = ((fp_t)dotid->pcompl) / fpcnt;
factor = scale ? factor : FP_NOUGHT;
#endif
log_record(dotid, id, EVENT_ASYNC_FRACTION, pcnt, 0);
}
}
void __t_dot_sched_async(int id, int idx, async_t *dotid, int p, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *result, int release) 
{
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
int wait;
int log = 0;
int pcnt;
int ready = 0;
float len = local_result[0];
int accept;
pthread_mutex_t *mutex = &dotid->mutex;
pthread_mutex_lock(mutex);
if ( dotid->consume != id ) {
if ( dotid->create == id ) {
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
} else {
BLAS_copy(bn, local_result, i_one, result, i_one);
dotid->create = id;
dotid->pcnt = 0;
}
if ( dotid->wait > 0 ) {
log = 1;
dotid->consume = id;
dotid->wait = 0;
pthread_cond_broadcast(&dotid->cond);
}
log_record(dotid, id, EVENT_PARTICIPATE, idx, len);
accept = 1;
} else {
log_record(dotid, id, EVENT_REFUSED, idx, len);
accept = 0;
}
pcnt = ++dotid->pcnt;
ready = pcnt == dotid->pcompl;	
dotid->ready += ready? 1: 0;
prof_add(&dotid->prof, idx, len);
prof_reg(&dotid->prof, idx, accept);
pthread_mutex_unlock(mutex);
if ( log && dotid->log ) {
#if 0
fp_t fpcnt = pcnt;
int scale = dotid->flags & ASYNC_SCALE;
fp_t factor = ((fp_t)dotid->pcompl) / fpcnt;
factor = scale ? factor : FP_NOUGHT;
#endif
}
}
void __t_dot_sched_async_release(int id, int idx, async_t *dotid, int p, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *result, int release) 
{
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
int wait;
int log = 0;
int pcnt;
int ready = 0;
float len = local_result[0];
int accept;
pthread_mutex_t *mutex = &dotid->mutex;
pthread_mutex_lock(mutex);
if ( dotid->consume != id && dotid->dot_control != id) {
if ( dotid->create == id ) {
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
} else {
BLAS_copy(bn, local_result, i_one, result, i_one);
dotid->create = id;
dotid->pcnt = 0;
}
if ( (dotid->pcnt+1) >= dotid->pcompl-release){
dotid->dot_control = id;
}
if ( dotid->wait > 0 && dotid->dot_control==id) {
log = 1;
dotid->consume = id;
dotid->wait = 0;
pthread_cond_broadcast(&dotid->cond);
}
log_record(dotid, id, EVENT_PARTICIPATE, idx, len);
accept = 1;
} else {
log_record(dotid, id, EVENT_REFUSED, idx, len);
if ( dotid->consume != id && dotid->wait > 0 ) {
log = 1;
dotid->consume = id;
dotid->wait = 0;
pthread_cond_broadcast(&dotid->cond);
}
accept = 0;
}
pcnt = ++dotid->pcnt;
ready = pcnt == dotid->pcompl;	
dotid->ready += ready? 1: 0;
prof_add(&dotid->prof, idx, len);
prof_reg(&dotid->prof, idx, accept);
pthread_mutex_unlock(mutex);
if ( log && dotid->log ) {
#if 0
fp_t fpcnt = pcnt;
int scale = dotid->flags & ASYNC_SCALE;
fp_t factor = ((fp_t)dotid->pcompl) / fpcnt;
factor = scale ? factor : FP_NOUGHT;
#endif
}
}
void __t_dot3_async(int id, int idx, async_t *sync, int p, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *r1, fp_t *r2, int release) 
{
fp_t lr1[bn];
fp_t lr2[bn];
int j;
for ( j=0; j<bn; ++j ) {
lr1[j] = BLAS_dot(bm, X, i_one, X, i_one);
lr2[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
float len = lr1[0] + lr2[0];
int log = 0;
int pcnt;
int ready = 0;
int accept;
pthread_mutex_t *mutex = &sync->mutex;
pthread_mutex_lock(mutex);
if ( sync->consume != id && sync->dot_control != id) {
if ( sync->create == id ) {
BLAS_axpy(bn, FP_ONE, lr1, i_one, r1, i_one);
BLAS_axpy(bn, FP_ONE, lr2, i_one, r2, i_one);
} else {
BLAS_copy(bn, lr1, i_one, r1, i_one);
BLAS_copy(bn, lr2, i_one, r2, i_one);
sync->create = id;
sync->pcnt = 0;
}
if ( (sync->pcnt+1) >= sync->pcompl-release){
sync->dot_control = id;
}
if ( sync->wait > 0 && sync->dot_control == id ) {
log = 1;
sync->consume = id;
sync->wait = 0;
pthread_cond_broadcast(&sync->cond);
}
accept = 1;
log_record(sync, id, EVENT_PARTICIPATE, idx, len);
} else {
log_record(sync, id, EVENT_REFUSED, idx, len);
if ( sync->wait > 0 && sync->dot_control == id ) {
log = 1;
sync->consume = id;
sync->wait = 0;
pthread_cond_broadcast(&sync->cond);
}
accept = 0;
}
pcnt = ++sync->pcnt;
ready = pcnt == sync->pcompl;	
sync->ready += ready? 1: 0;
prof_t *prof = &sync->prof;
if ( prof->s ) {
prof_add(&sync->prof, idx, len);
prof_reg(&sync->prof, idx, accept);
}
pthread_mutex_unlock(mutex);
if ( log ) {
#if 0
fp_t fpcnt = pcnt;
int scale = dotid->flags & ASYNC_SCALE;
fp_t factor = ((fp_t)dotid->pcompl) / fpcnt;
factor = scale ? factor : FP_NOUGHT;
log_record(dotid, id, EVENT_ASYNC_FRACTION, pcnt, 0);
#endif
}
}
void __t_dot_sched_async_concurrent(int id, int idx, async_t *dotid, int p, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *result, int release, int *bitmap) 
{
#pragma omp critical 
{
if ( bitmap[idx] ) {
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
if ( dotid->create == id ) {
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
} else {
BLAS_copy(bn, local_result, i_one, result, i_one);
dotid->create = id;
}
prof_add(&dotid->prof, idx, 1.0);
prof_reg(&dotid->prof, idx, 1);
} else {
prof_add(&dotid->prof, idx, 1.0);
prof_reg(&dotid->prof, idx, 0);
}
} 
}
#if 0
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
int wait;
int log = 0;
int pcnt;
int pcompl;
int ready = 0;
float len = local_result[0];
int accept;
#pragma omp critical 
{
pcompl = dotid->pcompl;
pcnt = dotid->pcnt;
if ( ! ( pcnt >= pcompl - release ) ) {
if ( dotid->create == id ) {
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
} else {
BLAS_copy(bn, local_result, i_one, result, i_one);
dotid->create = id;
}
accept = 1;
} else {
accept = 0;
}
pcnt++;
ready = pcnt == dotid->pcompl;	
dotid->ready += ready ? 1: 0;
dotid->pcnt = ready ? 0 : pcnt;
prof_add(&dotid->prof, idx, len);
prof_reg(&dotid->prof, idx, accept);
}  
#endif
