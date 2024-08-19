#include "task_dot.h"
#include "fpblas.h"
#include "fptype.h"
#include "blas.h"
#include "async_struct.h"
#include "task_log.h"
#define DRYRUN	0
#ifdef SINGLE_PRECISION 
#define __t_dot 			task_sdot
#define __t_dot_pure		task_sdot_pure
#define __t_dot3 			task_sdot3
#define __t_dot4 			task_sdot4
#else
#define __t_dot 			task_ddot
#define __t_dot_pure 		task_ddot_pure
#define __t_dot3 			task_ddot3
#define __t_dot4 			task_ddot4
#endif
void __t_dot(int id, int idx, async_t *sync, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *result) 
{
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
float len = local_result[0];
#pragma omp critical
{
int pcnt = sync->pcnt += 1;
if ( sync->create == id ) {
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
} else {
BLAS_copy(bn, local_result, i_one, result, i_one);
sync->create = id;
}
int ready = pcnt == sync->pcompl;
sync->pcnt = ready? 0: pcnt;
sync->ready += ready? 1: 0;
prof_t *prof = &sync->prof;
log_record(sync, id, EVENT_PARTICIPATE, idx, len);
prof_add(prof, idx, len);
} 
}
void __t_dot_pure(int p, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *result) 
{
fp_t local_result[bn];
int j;
for ( j=0; j<bn; ++j ) {
local_result[j] = BLAS_dot(bm, X, i_one, Y, i_one);
X += m;
Y += m;
}
#pragma omp critical
{
BLAS_axpy(bn, FP_ONE, local_result, i_one, result, i_one);
}
}
void __t_dot3(int id, int idx, async_t *sync, int bm, int bn, int m, int n, fp_t *X, fp_t *Y, fp_t *r1, fp_t *r2) 
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
pthread_mutex_t *mutex = &sync->mutex;
pthread_mutex_lock(mutex);
int pcnt = sync->pcnt += 1;
if ( sync->create == id ) {
BLAS_axpy(bn, FP_ONE, lr1, i_one, r1, i_one);
BLAS_axpy(bn, FP_ONE, lr2, i_one, r2, i_one);
} else {
BLAS_copy(bn, lr1, i_one, r1, i_one);
BLAS_copy(bn, lr2, i_one, r2, i_one);
sync->create = id;
sync->pcnt = 1;
}
int ready = pcnt == sync->pcompl;	
sync->pcnt = ready? 0: pcnt;	
sync->ready += ready? 1: 0;
prof_t *prof = &sync->prof;
log_record(sync, id, EVENT_PARTICIPATE, idx, len);
prof_add(prof, idx, len);
pthread_mutex_unlock(mutex);
}
void __t_dot4(int id, async_t *sync, int bm, int bn, int m, int n, fp_t *prevr1, fp_t *prevalpha, fp_t *X, fp_t *Y, fp_t *r1, fp_t *r2, fp_t *alpha) 
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
pthread_mutex_t *mutex = &sync->mutex;
pthread_mutex_lock(mutex);
int pcnt = sync->pcnt += 1;
if ( sync->create == id ) {
BLAS_axpy(bn, FP_ONE, lr1, i_one, r1, i_one);
BLAS_axpy(bn, FP_ONE, lr2, i_one, r2, i_one);
} else {
BLAS_copy(bn, lr1, i_one, r1, i_one);
BLAS_copy(bn, lr2, i_one, r2, i_one);
sync->create = id;
sync->pcnt = 1;
}
int ready = pcnt == sync->pcompl;	
if ( ready ) {
if ( prevr1[0] < 0 ) {
alpha[0] = r1[0] / r2[0] ;
} else {
alpha[0] = r1[0] / ( r2[0] - (r1[0] / prevr1[0]) * ( r1[0] / prevalpha[0] ) );
}
}
sync->pcnt = ready? 0: pcnt;	
sync->ready += ready? 1: 0;
pthread_mutex_unlock(mutex);
}
