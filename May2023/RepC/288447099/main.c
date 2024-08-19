#include "random.h"
#include "utils.h"
#include "partition.h"
#include "timing.h"
#include "validation.h"
#include "sylvester.h"
#include "robust.h"
#include "reference.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>
#include "omp.h"
int main(int argc, char **argv)
{
int m, n, tlsz;
double cmplx_ratio_A, cmplx_ratio_B;
unsigned int seed;
double *A, *B, *C, *X, *Y;
int ldA, ldB, ldC, ldX, ldY;
double sign;
memory_layout_t mem_layout;
if (argc != 9) {
printf("Usage %s n m tlsz cmplx-ratio-A cmplx-ratio-B sign "
"mem-layout seed\n", argv[0]);
printf("m:             Dimension of square matrix A\n");
printf("n:             Dimension of square matrix B\n");
printf("tlsz:          The tile size\n");
printf("cmplx-ratio-A: Ratio of complex/real eigenvalues of A\n");
printf("cmplx-ratio-B: Ratio of complex/real eigenvalues of B\n");
printf("sign:          1 or -1\n");
printf("mem-layout:    0=column major or 1=tile layout\n");
printf("seed:          Seed for the random number generator\n");
return EXIT_FAILURE;
}
m = atoi(argv[1]);
n = atoi(argv[2]);
tlsz = atoi(argv[3]);
cmplx_ratio_A = atof(argv[4]);
cmplx_ratio_B = atof(argv[5]);
sign = (double) atoi(argv[6]);
mem_layout = atoi(argv[7]);
seed = (unsigned int)atoi(argv[8]);
srand(seed);
assert(cmplx_ratio_A >= 0.0 && cmplx_ratio_A <= 1.0);
assert(cmplx_ratio_B >= 0.0 && cmplx_ratio_B <= 1.0);
assert(n > 0);
assert(m > 0);
assert(tlsz > 0);
assert(sign == 1.0 || sign == -1.0);
assert(mem_layout == COLUMN_MAJOR || mem_layout == TILE_LAYOUT);
printf("Configuration:\n");
#pragma omp parallel
#pragma omp single
printf("  OpenMP threads = %d\n", omp_get_num_threads());
printf("  m = %d\n", m);
printf("  n = %d\n", n);
printf("  tlsz = %d\n", tlsz);
printf("  cmplx-ratio-A = %.6lf\n", cmplx_ratio_A);
printf("  cmplx-ratio-B = %.6lf\n", cmplx_ratio_B);
printf("  Seed = %u\n", seed);
printf("  Used memory layout: ");
if (mem_layout == TILE_LAYOUT)
printf("tile layout\n");
else
printf("column major\n");
ldA = get_size_with_padding(m);
ldB = get_size_with_padding(n);
ldC = get_size_with_padding(m);
ldX = ldC;
ldY = ldC;
A = (double *) _mm_malloc((size_t)ldA * m * sizeof(double), ALIGNMENT);
B = (double *) _mm_malloc((size_t)ldB * n * sizeof(double), ALIGNMENT);
C = (double *) _mm_malloc((size_t)ldC * n * sizeof(double), ALIGNMENT);
X = (double *) _mm_malloc((size_t)ldX * n * sizeof(double), ALIGNMENT);
Y = (double *) _mm_malloc((size_t)ldY * n * sizeof(double), ALIGNMENT);
double *lambda_A, *lambda_B;
int *lambda_type_A, *lambda_type_B;
lambda_A = (double *) _mm_malloc(m * sizeof(double), ALIGNMENT);
lambda_B = (double *) _mm_malloc(n * sizeof(double), ALIGNMENT);
lambda_type_A = (int *) malloc(m * sizeof(int));
lambda_type_B = (int *) malloc(n * sizeof(int));
generate_eigenvalues(m, cmplx_ratio_A, lambda_A, lambda_type_A);
generate_eigenvalues(n, cmplx_ratio_B, lambda_B, lambda_type_B);
int num_tile_rows = (m + tlsz - 1) / tlsz;
int num_tile_cols = (n + tlsz - 1) / tlsz;
int *first_row = (int *) malloc((num_tile_rows + 1) * sizeof(int));
int *first_col = (int *) malloc((num_tile_cols + 1) * sizeof(int));
partitioning_t part_A = {.num_blk_rows = num_tile_rows,
.num_blk_cols = num_tile_rows,
.first_row = first_row,
.first_col = first_row};
partitioning_t part_B = {.num_blk_rows = num_tile_cols,
.num_blk_cols = num_tile_cols,
.first_row = first_col,
.first_col = first_col};
partitioning_t part_C = {.num_blk_rows = num_tile_rows,
.num_blk_cols = num_tile_cols,
.first_row = first_row,
.first_col = first_col};
partition(m, lambda_type_A, num_tile_rows, tlsz, first_row);
partition(n, lambda_type_B, num_tile_cols, tlsz, first_col);
double ***A_tiles = malloc(num_tile_rows * sizeof(double **));
for (int i = 0; i < num_tile_rows; i++) {
A_tiles[i] = malloc(num_tile_rows * sizeof(double *));
}
partition_matrix(A, ldA, mem_layout, &part_A, A_tiles);
double ***B_tiles = malloc(num_tile_cols * sizeof(double **));
for (int i = 0; i < num_tile_cols; i++) {
B_tiles[i] = malloc(num_tile_cols * sizeof(double *));
}
partition_matrix(B, ldB, mem_layout, &part_B, B_tiles);
double ***C_tiles = malloc(num_tile_rows * sizeof(double **));
for (int i = 0; i < num_tile_rows; i++) {
C_tiles[i] = malloc(num_tile_cols * sizeof(double *));
}
partition_matrix(C, ldC, mem_layout, &part_C, C_tiles);
double ***X_tiles = malloc(num_tile_rows * sizeof(double **));
for (int i = 0; i < num_tile_rows; i++) {
X_tiles[i] = malloc(num_tile_cols * sizeof(double *));
}
partition_matrix(X, ldX, mem_layout, &part_C, X_tiles);
double ***Y_tiles = malloc(num_tile_rows * sizeof(double **));
for (int i = 0; i < num_tile_rows; i++) {
Y_tiles[i] = malloc(num_tile_cols * sizeof(double *));
}
partition_matrix(Y, ldY, mem_layout, &part_C, Y_tiles);
printf("Generate A...\n");
generate_upper_quasi_triangular_matrix(
A_tiles, ldA, &part_A, lambda_A, lambda_type_A, mem_layout);
if (mem_layout == COLUMN_MAJOR) {
int err = validate_quasi_triangular_shape(m, A, ldA, UPPER_TRIANGULAR);
if (err != 0) {
printf("ERROR: Invalid input matrix A. Bye.\n");
return EXIT_FAILURE;
}
}
printf("Generate B...\n");
generate_upper_quasi_triangular_matrix(
B_tiles, ldB, &part_B, lambda_B, lambda_type_B, mem_layout);
if (mem_layout == COLUMN_MAJOR) {
int err = validate_quasi_triangular_shape(n, B, ldB, UPPER_TRIANGULAR);
if (err != 0) {
printf("ERROR: Invalid input matrix B. Bye.\n");
return EXIT_FAILURE;
}
err = validate_spectra(sign, m, lambda_A, lambda_type_A,
n, lambda_B, lambda_type_B);
if (err != 0) {
printf("ERROR: Matrices have common eigenvalues. Bye.\n");
return EXIT_FAILURE;
}
}
printf("Generate C...\n");
generate_dense_matrix(C_tiles, ldC, &part_C, mem_layout);
printf("Save a copy of C...\n");
copy_matrix(C_tiles, ldC, X_tiles, ldX, &part_C, mem_layout);
copy_matrix(C_tiles, ldC, Y_tiles, ldY, &part_C, mem_layout);
scaling_t scale;
if (sign == 1.0)
printf("Solve AX + XB = C\n");
else
printf("Solve AX - XB = C\n");
double tm_start = get_time();
solve_tiled_sylvester(sign, A_tiles, ldA, B_tiles, ldB,
X_tiles, ldX, &part_C, &scale, mem_layout);
printf("Execution time = %.2f s.\n", get_time() - tm_start);
printf("Validate...\n");
validate_tiled(sign,
A_tiles, ldA, &part_A, 
B_tiles, ldB, &part_B, 
C_tiles, ldC, &part_C,
X_tiles, ldX,
scale,
mem_layout);
#ifdef INTSCALING
printf("Scale = 2^%d\n", scale);
#else
printf("Scale = %.6e\n", convert_scaling(scale));
#endif
long long flops = (long long)m * m * n + (long long)n * n * m;
printf("Flops = %lld\n", flops);
if (mem_layout == COLUMN_MAJOR) {
printf("Compute reference solution with LAPACK...\n");
solve_sylvester_dtrsyl('N', 'N', sign, A_tiles, ldA, B_tiles, ldB,
Y_tiles, ldY, &part_C);
}
_mm_free(A);
_mm_free(B);
_mm_free(C);
_mm_free(X);
_mm_free(Y);
_mm_free(lambda_A);
_mm_free(lambda_B);
free(lambda_type_A);
free(lambda_type_B);
free(first_row);
free(first_col);
for (int i = 0; i < num_tile_rows; i++) {
free(A_tiles[i]);
free(C_tiles[i]);
free(X_tiles[i]);
free(Y_tiles[i]);
}
for (int i = 0; i < num_tile_cols; i++) {
free(B_tiles[i]);
}
free(A_tiles);
free(B_tiles);
free(C_tiles);
free(X_tiles);
free(Y_tiles);
return EXIT_SUCCESS;
}
