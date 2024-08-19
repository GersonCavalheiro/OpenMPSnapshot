#include "utils.h"
#include "defines.h"
#include "norm.h"
#include "update-task.h"
#include "robust.h"
#include <stdlib.h>
#include <stdio.h>
#include <mm_malloc.h>
#include <string.h>
#include <math.h>
void update(
int m, int n, int k,
omp_lock_t *lock,
const double alpha, double *restrict const Ain, int ldAin,
const double ainnorm, const scaling_t ainscale,
double *restrict const B, int ldB, const double bnorm,
double *restrict const C, int ldC, double *restrict const cnorm, 
scaling_t *restrict const cscale)
{
#ifndef NDEBUG
printf("update (%dx%d)(%dx%d)\n", m, k, k, n);
printf("update thread id = %d\n", omp_get_thread_num());
#endif
scaling_t cscaling = *cscale;
scaling_t zeta;
double *A;
double anorm;
int rescale_A = 0;
int rescale_C = 0;
while (!omp_test_lock(lock)) {
#pragma omp taskyield
;
}
anorm = ainnorm;
*cnorm = matrix_infnorm(m, n, C, ldC);
if (cscaling < ainscale) {
const double s = compute_upscaling(cscaling, ainscale);
rescale_A = 1;
anorm = s * ainnorm;
}
else if (ainscale < cscaling) {
const double s = compute_upscaling(ainscale, cscaling);
rescale_C = 1;
*cnorm = s * (*cnorm);
}
else {
}
zeta = protect_update(anorm, bnorm, *cnorm);
#ifdef INTSCALING
if (zeta != 0) {
rescale_A = 1;
rescale_C = 1;
}
#else
if (zeta != 1.0) {
rescale_A = 1;
rescale_C = 1;
}
#endif
if (rescale_A) {
A = (double *) _mm_malloc((size_t)ldAin * k * sizeof(double), ALIGNMENT);
if (cscaling < ainscale) {
double s = compute_combined_upscaling(cscaling, ainscale, zeta);
for (int j = 0; j < k; j++)
for (int i = 0; i < m; i++)
A[i + ldAin * j] = s * Ain[i + ldAin * j];
}
else if (ainscale < cscaling) {
double s = convert_scaling(zeta);
for (int j = 0; j < k; j++)
for (int i = 0; i < m; i++)
A[i + ldAin * j] = s * Ain[i + ldAin * j];
}
}
else {
A = Ain;
}
if (rescale_C) {
if (cscaling < ainscale) {
const double s = convert_scaling(zeta);
for (int j = 0; j < n; j++)
for (int i = 0; i < m; i++)
C[i + j * ldC] = s * C[i + j * ldC];
}
else if (ainscale < cscaling) {
const double s = compute_combined_upscaling(ainscale, cscaling, zeta);
for (int j = 0; j < n; j++)
for (int i = 0; i < m; i++)
C[i + j * ldC] = s * C[i + j * ldC];
}
else {
const double s = convert_scaling(zeta);
for (int j = 0; j < n; j++)
for (int i = 0; i < m; i++)
C[i + j * ldC] = s * C[i + j * ldC];
}
}
#ifdef INTSCALING
*cscale = min(cscaling, ainscale) + zeta;
#else
*cscale = minf(cscaling, ainscale) * zeta;
#endif
dgemm('N', 'N',
m, n, k,
-alpha, A, ldAin,
B, ldB,
1.0, C, ldC);
omp_unset_lock(lock);
if (rescale_A) {
_mm_free(A);
}
}
