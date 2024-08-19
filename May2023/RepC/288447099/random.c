#include "random.h"
#include "norm.h"
#include "utils.h"
#include "defines.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#ifdef DISTRIBUTED
#include "mpi.h"
#endif
int random_integer (int low, int high)
{
int size = high - low + 1;
double x = random_double (0, 1);
int k = (int) (x * size);
if (k == size) {
--k;
}
return low + k;
}
double random_double (double low, double high)
{
double x = (double) rand () / RAND_MAX;
return low + x * (high - low);
}
static void generate_dense_block(int n, int m, double *restrict const A, int ld)
{
const double low = 0.1;
const double high = 1.0;
for (int i = 0; i < n; i++) {
for (int j = 0; j < m; j++) {
A[i + (size_t)ld *j] = random_double(low, high);
}
}
}
void generate_dense_matrix(
double ***A_blocks, int ld,
partitioning_t *p,
memory_layout_t mem_layout)
{
int *first_row = p->first_row;
int *first_col = p->first_col;
int num_blk_rows = p->num_blk_rows;
int num_blk_cols = p->num_blk_cols;
#pragma omp parallel
#pragma omp single nowait
for (int i = 0; i < num_blk_rows; i++) {
for (int j = 0; j < num_blk_cols; j++) {
#pragma omp task
{
const int num_rows = first_row[i + 1] - first_row[i];
const int num_cols = first_col[j + 1] - first_col[j];
if (mem_layout == COLUMN_MAJOR)
generate_dense_block(num_rows, num_cols, A_blocks[i][j], ld);
else 
generate_dense_block(num_rows, num_cols, A_blocks[i][j], num_rows);
}
}
}
}
static void generate_upper_quasi_triangular_block(
int n, int ld, double *restrict const T,
const double *restrict const lambda, const int *restrict const lambda_type)
{
#define T(i,j) T[(i) + ld * (j)]
const double low = 0.1;
const double high = 1.0;
for (int j = 0; j < n; j++) {
for (int i = 0; i <= j; i++) {
T(i,j) = random_double(low, high);
}
for (int i = j + 1; i < n; i++) {
T(i,j) = 0.0;
}
}
for (int j = 0; j < n; j++) {
if (lambda_type[j] == REAL) {
T(j,j) = lambda[j];
}
else {
const double a = lambda[j];
const double b = lambda[j+1];
T(j  ,j) = a;  T(j  ,j+1) = -b;
T(j+1,j) = b;  T(j+1,j+1) = a;
j++;
}
}
#undef T
}
void generate_upper_quasi_triangular_matrix(
double ***T_blocks, int ld,
partitioning_t *p,
const double *restrict const lambda, const int *restrict const lambda_type,
memory_layout_t mem_layout)
{
int num_blk_rows = p->num_blk_rows;
#ifndef NDEBUG
int num_blk_cols = p->num_blk_cols;
#endif
int *first_row = p->first_row;
int *first_col = p->first_col;
int first_blk = 0;
assert(num_blk_rows == num_blk_cols);
int num_blks = num_blk_rows;
{
for (int i = first_blk; i < first_blk + num_blks; i++) {
for (int j = i + 1; j < first_blk + num_blks; j++) {
const int num_rows = first_row[i + 1] - first_row[i];
const int num_cols = first_col[j + 1] - first_col[j];
if (mem_layout == COLUMN_MAJOR)
generate_dense_block(num_rows, num_cols, T_blocks[i][j], ld);
else 
generate_dense_block(num_rows, num_cols, T_blocks[i][j], num_rows);
}
}
for (int j = first_blk; j < first_blk + num_blks; j++) {
const int num_cols = first_col[j + 1] - first_col[j];
if (mem_layout == COLUMN_MAJOR)
generate_upper_quasi_triangular_block(
num_cols, ld, T_blocks[j][j], 
&lambda[first_col[j]], &lambda_type[first_col[j]]);
else
generate_upper_quasi_triangular_block(
num_cols, num_cols, T_blocks[j][j], 
&lambda[first_col[j]], &lambda_type[first_col[j]]);
}
for (int i = first_blk; i < first_blk + num_blks; i++) {
for (int j = first_blk; j < i; j++) {
const int num_rows = first_row[i + 1] - first_row[i];
const int num_cols = first_col[j + 1] - first_col[j];
int ldT;
if (mem_layout == TILE_LAYOUT)
ldT = num_rows;
else
ldT = ld;
set_zero(num_rows, num_cols, T_blocks[i][j], ldT);
}
}
}
}
void generate_eigenvalues(const int n, double complex_ratio,
double *restrict const lambda, int *restrict const lambda_type)
{
int complex_count = complex_ratio * n / 2;
int real_count = n - 2 * complex_count;
int spaces[complex_count+1];
for (int i = 0; i < complex_count+1; i++)
spaces[i] = 0;
for (int i = 0; i < real_count; i++)
spaces[rand() % (complex_count+1)]++;
for (int i = 0; i < n; i++) {
lambda[i] =  n + 1.0 + i;
lambda_type[i] = REAL;
}
int i = 0;
for (int j = 0; j < complex_count; j++) {
i += spaces[j];
int grid_height = sqrt(2*complex_count)/2;
if (grid_height == 0) {
lambda[i] = n + 0.5 + j;
lambda[i+1] = n + 1.5 + j;
lambda_type[i] = REAL;
lambda_type[i+1] = REAL;
i += 2;
}
else {
double real = j / grid_height - grid_height + 0.5;
double imag = j % grid_height + 1.0;
lambda[i] = n + real;
lambda[i+1] = imag;
lambda_type[i] = CMPLX;
lambda_type[i+1] = CMPLX;
i += 2;
}
}
}
