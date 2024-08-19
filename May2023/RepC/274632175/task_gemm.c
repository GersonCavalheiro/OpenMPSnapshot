#include "task_gemm.h"
#include "fptype.h"
#include "fpblas.h"
#include "blas.h"
#include "async_struct.h"
#include "selfsched.h"
#include "prof.h"
#include "task_log.h"
#ifdef DOUBLE_PRECISION
#define __t_gemm			task_dgemm
#define __t_gemmcg			task_dgemmcg
#define __t_gemmcg_release	task_dgemmcg_release
#define __t_gemmcg_switch	task_dgemmcg_switch
#else
#define __t_gemm			task_sgemm
#define __t_gemmcg			task_sgemmcg
#define __t_gemmcg_release	task_sgemmcg_release
#define __t_gemmcg_switch	task_sgemmcg_switch
#endif
void __t_gemmcg(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, fp_t alpha, fp_t *A, int lda, fp_t *B, int ldb, fp_t beta, fp_t *C, int ldc, int p, int idx) 
{
#if 0
if ( idx != -1 ) {
pthread_mutex_t *mutex = &sync->mutex;
pthread_mutex_lock(mutex);
#pragma omp critical 
{
log_record(sync, it, EVENT_PRIORITY, idx, (float)p);
}
pthread_mutex_unlock(mutex);
}
#endif
BLAS_gemm(transa, transb, bm, bn, bk, alpha, A, lda, B, ldb, beta, C, ldc);
}
void __t_gemm(ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, fp_t alpha, fp_t *A, int lda, fp_t *B, int ldb, fp_t beta, fp_t *C, int ldc, int p) 
{
BLAS_gemm(transa, transb, bm, bn, bk, alpha, A, lda, B, ldb, beta, C, ldc);
}
void __t_gemmcg_release(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, fp_t alpha, fp_t *A, int lda, fp_t *B, int ldb, fp_t beta, fp_t *C, int ldc, int p, int idx, int release, int *bitmap) 
{
if ( idx >= 0 ) {
int i;
for ( i = 0; i < bm; i++ ) {
C[i] = 0.0;
}
}
int pcompl = sync->pcompl;
int pcnt;
#pragma omp critical
{
if ( idx >= 0 ) {
pcnt = sync->pcnt++;
if ( sync->pcnt == sync->pcompl ) {
sync->pcnt = 0;
}
} else {
pcnt = sync->pcnt;
}
}
if (  pcnt < pcompl - release ) {
*bitmap = 1;
BLAS_gemm(transa, transb, bm, bn, bk, alpha, A, lda, B, ldb, beta, C, ldc);
} else {
*bitmap = 0;
}
}
void __t_gemmcg_switch(int it, async_t *sync, ompssblas_t transa, ompssblas_t transb, int bm, int bn, int bk, fp_t alpha, fp_t *A, int lda, fp_t *B, int ldb, fp_t beta, fp_t *C, int ldc, int p, int idx, int release, int *bitmap, int on)
{
if ( idx >= 0 ) {
int i;
for ( i = 0; i < bm; i++ ) {
C[i] = 0.0;
}
}
int pcompl = sync->pcompl;
int pcnt;
#pragma omp critical
{
if ( idx >= 0 ) {
pcnt = sync->pcnt++;
if ( sync->pcnt == sync->pcompl ) {
sync->pcnt = 0;
}
} else {
pcnt = sync->pcnt;
}
}
if ( on ) {
*bitmap = 1;
BLAS_gemm(transa, transb, bm, bn, bk, alpha, A, lda, B, ldb, beta, C, ldc);
} else {
*bitmap = 0;
}
}
