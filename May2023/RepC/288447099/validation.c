#include "validation.h"
#include "utils.h"
#include "partition.h"
#include "defines.h"
#include "norm.h"
#include <mm_malloc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "omp.h"
static void tiled_dgemm(
double alpha,
double ***A_tiles, int ldA, partitioning_t *part_A,
double ***B_tiles, int ldB, partitioning_t *part_B,
double beta,
double ***C_tiles, int ldC, partitioning_t *part_C,
memory_layout_t mem_layout)
{
#pragma omp parallel
#pragma omp single nowait
for (int i = 0; i < part_C->num_blk_rows; i++) {
for (int j = 0; j < part_C->num_blk_cols; j++) {
#pragma omp task
{
const int m = part_C->first_row[i + 1] - part_C->first_row[i];
const int n = part_C->first_col[j + 1] - part_C->first_col[j];
for (int l = 0; l < part_A->num_blk_cols; l++) {
int k = part_A->first_col[l + 1] - part_A->first_col[l];
if (mem_layout == COLUMN_MAJOR)
dgemm('N', 'N', m, n, k,
alpha, A_tiles[i][l], ldA,
B_tiles[l][j], ldB,
beta, C_tiles[i][j], ldC);
else 
dgemm('N', 'N', m, n, k,
alpha, A_tiles[i][l], m,
B_tiles[l][j], k,
beta, C_tiles[i][j], m);
}
}
}
}
}
void validate(
double sgn,
int m, int n,
const double *restrict const A, int ldA,
const double *restrict const B, int ldB,
double *restrict C, int ldC,
double *restrict X, int ldX,
const scaling_t scale)
{
double normA, normB, normC, normX, normR;
normA = dlange('F', m, m, A, ldA);
normB = dlange('F', n, n, B, ldB);
normC = convert_scaling(scale) * dlange('F', m, n, C, ldC);
normX = dlange('F', m, n, X, ldX);
double beta;
#ifdef INTSCALING
beta = ldexp(1.0, scale);
#else
beta = scale;
#endif
dgemm('N', 'N', m, n, m, -1.0, A, ldA, X, ldX, beta, C, ldC);
dgemm('N', 'N', m, n, n, -1.0 * sgn, X, ldX, B, ldB, 1.0, C, ldC);
normR = dlange('F', m, n, C, ldC);
printf("|| scale * C - (A * X + sgn * X * B)||_oo = %.6e\n", normR);
double err = normR / ((normA + normB) * normX + normC);
printf("|| scale * C - (A * X + sgn * X * B)||_F / "
"((||A||_F + ||B||_F) * ||X||_F + ||scale * C||_F) = %.6e\n", err);
printf("LAPACK normA = %.6e\n", normA);
printf("LAPACK normB = %.6e\n", normB);
printf("LAPACK normX = %.6e\n", normX);
printf("LAPACK normC = %.6e\n", normC);
printf("Beware that mass can be moved between scale and normX, yielding "
"different relative residuals. Compare normX and scale to judge "
"if the relative residual is correct.\n");
}
void validate_tiled(
double sgn,
double ***A_tiles, int ldA, partitioning_t *p_A,
double ***B_tiles, int ldB, partitioning_t *p_B,
double ***C_tiles, int ldC, partitioning_t *p_C,
double ***X_tiles, int ldX,
const scaling_t scale,
memory_layout_t mem_layout)
{
double db_scale = convert_scaling(scale);
double normA, normB, normC, normX, normR;
normA = tiled_matrix_frobeniusnorm(A_tiles, ldA, p_A, mem_layout);
normB = tiled_matrix_frobeniusnorm(B_tiles, ldB, p_B, mem_layout);
normC = convert_scaling(scale) * 
tiled_matrix_frobeniusnorm(C_tiles, ldC, p_C, mem_layout);
normX = tiled_matrix_frobeniusnorm(X_tiles, ldX, p_C, mem_layout);
tiled_dgemm(-1.0, A_tiles, ldA, p_A, X_tiles, ldX, p_C,
db_scale, C_tiles, ldC, p_C, mem_layout);
tiled_dgemm(-1.0 * sgn, X_tiles, ldX, p_C, B_tiles, ldB, p_B,
1.0, C_tiles, ldC, p_C, mem_layout);
normR = tiled_matrix_frobeniusnorm(C_tiles, ldC, p_C, mem_layout);
printf("|| scale * C - (A * X + sgn * X * B)||_oo = %.6e\n", normR);
double err = normR / ((normA + normB) * normX + normC);
printf("|| scale * C - (A * X + sgn * X * B)||_F / "
"((||A||_F + ||B||_F) * ||X||_F + ||scale * C||_F) = %.6e\n", err);
printf("normA = %.6e\n", normA);
printf("normB = %.6e\n", normB);
printf("normX = %.6e\n", normX);
printf("normC = %.6e\n", normC);
printf("Beware that mass can be moved between scale and normX, yielding "
"different relative residuals. Compare normX and scale to judge "
"if the relative residual is correct.\n");
}
int validate_quasi_triangular_shape(
int n, const double *restrict const A, int ldA, matrix_desc_t type)
{
switch (type) {
case UPPER_TRIANGULAR:
for (int i = 1; i < n - 1; i++) {
if ((A[(i + 1) + ldA * i] != 0.0) && (A[i + ldA * (i - 1)] != 0.0)) {
return -1;
}
}
break;
case LOWER_TRIANGULAR:
for (int i = 1; i < n - 1; i++) {
if ((A[i + ldA * (i + 1)] != 0.0) && (A[i - 1 + ldA * i] != 0.0)) {
return -1;
}
}
break;
}
return 0;
}
int validate_spectra(double sgn,
int m, double *lambda_A, int *lambda_type_A,
int n, double *lambda_B, int *lambda_type_B)
{
for (int i = 0; i < m; i++) {
if (lambda_type_A[i] == CMPLX) {
const double real = lambda_A[i];
const double imag = lambda_A[i + 1];
for (int j = 0; j < n; j++) {
if (lambda_type_B[j] == CMPLX) {
if (-sgn * lambda_B[j] == real && 
-sgn * lambda_B[j + 1] == imag)
return -1;
j++;
}
}
i++;
}
else { 
const double real = lambda_A[i];
for (int j = 0; j < n; j++) {
if (lambda_type_B[j] == REAL) {
if (real == -sgn * lambda_B[j]) {
return -1;
}
}
}
}
}
return 0;
}
