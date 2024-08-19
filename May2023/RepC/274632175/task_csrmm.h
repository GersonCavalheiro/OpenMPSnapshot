#ifndef __TASK_CSRMM_H__
#define __TASK_CSRMM_H__
#include "hb.h"
#include "selfsched.h"
#include "async_struct.h"
#ifdef SINGLE_PRECISION
#define TASK_CSRMM_SCHED 	task_scsrmm_sched
#define TASK_CSRMV		 	task_scsrmv
#define TASK_CSRMV_PRIOR	task_scsrmv_prior
#define TASK_CSRMV_RELEASE	task_scsrmv_release
#define TASK_CSRMV_SWITCH	task_scsrmv_switch
#define TASK_CSRSM			task_scsrsm
#define TASK_CSRMV_COMB		task_scsrmv_comb
#define TASK_CSRMV_SWITCH0	task_scsrmv_switch0
#else
#define TASK_CSRMM_SCHED 	task_dcsrmm_sched
#define TASK_CSRMV		 	task_dcsrmv
#define TASK_CSRMV_PRIOR	task_dcsrmv_prior
#define TASK_CSRMV_RELEASE	task_dcsrmv_release
#define TASK_CSRMV_SWITCH	task_dcsrmv_switch
#define TASK_CSRSM			task_dcsrsm
#define TASK_CSRMV_COMB		task_dcsrmv_comb
#define TASK_CSRMV_SWITCH0	task_dcsrmv_switch0
#endif
#define LIBSBBLAS_EXPORT __attribute__((__visibility__("default")))
#pragma omp task in([d*c]B) inout([b*c]C)
extern LIBSBBLAS_EXPORT void task_scsrmm_sched(int cnt, int b, int c, int d, float alpha, int *vptr, int *vpos, hbmat_t **vval, float *B, int ldb, float beta, float *C, int ldc, selfsched_t *sched);
#pragma omp task in([d*c]B) inout([b*c]C)
extern LIBSBBLAS_EXPORT void task_dcsrmm_sched(int cnt, int b, int c, int d, double alpha, int *vptr, int *vpos, hbmat_t **vval, double *B, int ldb, double beta, double *C, int ldc, selfsched_t *sched);
#pragma omp task in([m]B) inout([bs]C) priority(p) no_copy_deps label(scsrmv)
extern LIBSBBLAS_EXPORT void task_scsrmv(int p, char *trans, int bs, int m, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *B, float beta, float *C);
#pragma omp task in([m]B) inout([bs]C) priority(p) no_copy_deps label(dcsrmv)
extern LIBSBBLAS_EXPORT void task_dcsrmv(int p, char *trans, int bs, int m, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *B, double beta, double *C);
#pragma omp task in([m]B) inout([bs]C) priority(p) no_copy_deps label(scsrmv_prior)
extern LIBSBBLAS_EXPORT void task_scsrmv_prior(int p, char *trans, int bs, int m, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *B, float beta, float *C);
#pragma omp task in([m]B) inout([bs]C) priority(p) no_copy_deps label(dcsrmv_prior)
extern LIBSBBLAS_EXPORT void task_dcsrmv_prior(int p, char *trans, int bs, int m, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *B, double beta, double *C);
#pragma omp task in([m]B) inout([bs]C) priority(p) no_copy_deps label(scsrmv_release)
extern LIBSBBLAS_EXPORT void task_scsrmv_release(async_t *sync, int idx, int p, char *trans, int bs, int m, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *B, float beta, float *C, int release, int *bitmap);
#pragma omp task in([m]B) inout([bs]C) priority(p) no_copy_deps label(dcsrmv_release)
extern LIBSBBLAS_EXPORT void task_dcsrmv_release(async_t *sync, int idx, int p, char *trans, int bs, int m, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *B, double beta, double *C, int release, int *bitmap);
#pragma omp task in([m]B) inout([bs]C) no_copy_deps label(scsrmv_switch)
extern LIBSBBLAS_EXPORT void task_scsrmv_switch(async_t *sync, int idx, int p, char *trans, int bs, int m, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *B, float beta, float *C, int release, int *bitmap, int on);
#pragma omp task in([m]B) inout([bs]C) no_copy_deps label(dcsrmv_switch)
extern LIBSBBLAS_EXPORT void task_dcsrmv_switch(async_t *sync, int idx, int p, char *trans, int bs, int m, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *B, double beta, double *C, int release, int *bitmap, int on);
#pragma omp task in([m]b) out([m]x) priority(p) no_copy_deps label(scsrsm)
extern LIBSBBLAS_EXPORT void task_scsrsm(int p, char *trans, int m, int n, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *b, int ldb, float *x, int ldx);
#pragma omp task in([m]b) out([m]x) priority(p) no_copy_deps label(dcsrsm)
extern LIBSBBLAS_EXPORT void task_dcsrsm(int p, char *trans, int m, int n, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *b, int ldb, double *x, int ldx);
#pragma omp task in([m]B) concurrent([1]result) no_copy_deps label(scsrmv_comb)
extern LIBSBBLAS_EXPORT void task_scsrmv_comb(async_t *sync, int id, int idx, char *trans, int bs, int m, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *B, float beta, float *C, float *result, int on);
#pragma omp task in([m]B) concurrent([1]result) no_copy_deps label(dcsrmv_comb)
extern LIBSBBLAS_EXPORT void task_dcsrmv_comb(async_t *sync, int id, int idx, char *trans, int bs, int m, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *B, double beta, double *C, double *result, int on);
#pragma omp task in([m]B) inout([bs]C) no_copy_deps label(scsrmv_switch0)
extern LIBSBBLAS_EXPORT void task_scsrmv_switch0(int idx, char *trans, int bs, int m, float alpha, char *matdescra, float *vval, int *vpos, int *vptr, float *B, float beta, float *C, int on);
#pragma omp task in([m]B) inout([bs]C) no_copy_deps label(dcsrmv_switch0)
extern LIBSBBLAS_EXPORT void task_dcsrmv_switch0(int idx, char *trans, int bs, int m, double alpha, char *matdescra, double *vval, int *vpos, int *vptr, double *B, double beta, double *C, int on);
#endif 
