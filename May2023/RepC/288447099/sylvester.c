#include "sylvester.h"
#include "syl-task.h"
#include "update-task.h"
#include "typedefs.h"
#include "majorant.h"
#include "utils.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
static void init_2D_lock_grid(int m, int n, omp_lock_t *lock)
{
#define lock(i,j) lock[(i) + m * (j)]
for (int j = 0; j < n; j++) {
for (int i = 0; i < m; i++) {
omp_init_lock(&lock(i,j));
}
}
#undef lock
}
static void destroy_2D_lock_grid(int m, int n, omp_lock_t *lock)
{
#define lock(i,j) lock[(i) + m * (j)]
for (int j = 0; j < n; j++) {
for (int i = 0; i < m; i++) {
omp_destroy_lock(&lock(i,j));
}
}
#undef lock
}
void solve_tiled_sylvester(
double sgn,
double ***A_tiles, int ldA,
double ***B_tiles, int ldB,
double ***C_tiles, int ldC,
partitioning_t *p,
scaling_t *scale,
memory_layout_t mem_layout)
{
assert(sgn == 1.0 || sgn == -1.0);
int num_tile_rows = p->num_blk_rows;
int num_tile_cols = p->num_blk_cols;
int *first_row = p->first_row;
int *first_col = p->first_col;
const double eps = DBL_EPSILON / 2;
omp_lock_t lock[num_tile_rows][num_tile_cols];
init_2D_lock_grid(num_tile_rows, num_tile_cols, &lock[0][0]);
const int m = first_row[num_tile_rows];
const int n = first_col[num_tile_cols];
scaling_t *scales;
scales = (scaling_t *) malloc(
(size_t)num_tile_rows * num_tile_cols * sizeof(scaling_t));
init_scaling_factor(num_tile_rows * num_tile_cols, scales);
#define scales(tlrow, tlcol) scales[(tlrow) + (tlcol) * num_tile_rows]
double *A_norms, *B_norms, *C_norms;
A_norms = (double *) malloc(num_tile_rows * num_tile_rows * sizeof(double));
B_norms = (double *) malloc(num_tile_cols * num_tile_cols * sizeof(double));
C_norms = (double *) malloc(num_tile_rows * num_tile_cols * sizeof(double));
memset(A_norms, 0.0, num_tile_rows * num_tile_rows * sizeof(double));
memset(B_norms, 0.0, num_tile_cols * num_tile_cols * sizeof(double));
#define A_norms(tilerow,tilecol) A_norms[(tilerow) + (tilecol) * num_tile_rows]
#define B_norms(tilerow,tilecol) B_norms[(tilerow) + (tilecol) * num_tile_cols]
#define C_norms(tilerow,tilecol) C_norms[(tilerow) + (tilecol) * num_tile_rows]
#pragma omp parallel
#pragma omp single nowait
{
bound_triangular_matrix(A_tiles, ldA, A_norms, num_tile_rows, first_row,
UPPER_TRIANGULAR, mem_layout);
bound_triangular_matrix(B_tiles, ldB, B_norms, num_tile_cols, first_col,
UPPER_TRIANGULAR, mem_layout);
}
double smin = (DBL_MIN * (double)m) * ((double)n / eps);
{
double A_ub = 0.0;
for (int i = 0; i < num_tile_rows; i++) {
for (int j = 0; j < num_tile_rows; j++) {
A_ub = fmax(A_ub, A_norms(i,j));
}
}
double B_ub = 0.0;
for (int i = 0; i < num_tile_cols; i++) {
for (int j = 0; j < num_tile_cols; j++) {
B_ub = fmax(B_ub, B_norms(i,j));
}
}
smin = fmax(smin, fmax(eps * A_ub, eps * B_ub));
}
#pragma omp parallel
#pragma omp single nowait
for (int k = num_tile_rows - 1; k >= 0; k--) {
for (int l = 0; l < num_tile_cols; l++) {
#pragma omp task depend(inout:C_tiles[k][l]) depend(in:C_tiles[k+1:num_tile_rows][l]) depend(in:C_tiles[k][0:l-1]) shared(C_norms) shared(scales)
{
int num_rows = first_row[k + 1] - first_row[k];
int num_cols = first_col[l + 1] - first_col[l];
if (mem_layout == COLUMN_MAJOR) {
blocked_syl(sgn, num_rows, num_cols,
A_tiles[k][k], ldA, A_norms(k,k),
B_tiles[l][l], ldB, B_norms(l,l),
C_tiles[k][l], ldC,
&C_norms(k,l), &scales(k,l),
smin);
}
else { 
blocked_syl(sgn, num_rows, num_cols,
A_tiles[k][k], num_rows, A_norms(k,k),
B_tiles[l][l], num_cols, B_norms(l,l),
C_tiles[k][l], num_rows,
&C_norms(k,l), &scales(k,l),
smin);
}
}
for (int i = k - 1; i >= 0; i--) {
#pragma omp task depend(in:C_tiles[k][l]) depend(inout:C_tiles[i][l]) shared(lock) shared(C_norms)
{
const int num_rows = first_row[i + 1] - first_row[i];
const int num_cols = first_col[l + 1] - first_col[l];
int num_inner = first_row[k + 1] - first_row[k];
if (mem_layout == COLUMN_MAJOR)
update(num_rows, num_cols, num_inner, &lock[i][l], 1.0,
A_tiles[i][k], ldA, A_norms(i,k),
scales(k,l), C_tiles[k][l], ldC, C_norms(k,l),
C_tiles[i][l], ldC, &C_norms(i,l), &scales(i,l));
else 
update(num_rows, num_cols, num_inner, &lock[i][l], 1.0,
A_tiles[i][k], num_rows, A_norms(i,k),
scales(k,l), C_tiles[k][l], num_inner, C_norms(k,l),
C_tiles[i][l], num_rows, &C_norms(i,l), &scales(i,l));
}
}
for (int j = l + 1; j < num_tile_cols; j++) {
#pragma omp task depend(in:C_tiles[k][l]) depend(inout:C_tiles[k][j]) shared(lock)
{
const int num_rows = first_row[k + 1] - first_row[k];
const int num_cols = first_col[j + 1] - first_col[j];
const int num_inner = first_col[l + 1] - first_col[l];
if (mem_layout == COLUMN_MAJOR)
update(num_rows, num_cols, num_inner, &lock[k][j], sgn,
C_tiles[k][l], ldC, C_norms(k,l), scales(k,l),
B_tiles[l][j], ldB, B_norms(l,j),
C_tiles[k][j], ldC, &C_norms(k,j), &scales(k,j));
else 
update(num_rows, num_cols, num_inner, &lock[k][j], sgn,
C_tiles[k][l], num_rows, C_norms(k,l), scales(k,l),
B_tiles[l][j], num_inner, B_norms(l,j),
C_tiles[k][j], num_rows, &C_norms(k,j), &scales(k,j));
}
}
}
}
*scale = min_element(num_tile_rows * num_tile_cols, scales);
#ifndef INTSCALING
if (*scale == 0.0) {
printf("ERROR: The scaling was flushed to zero. The result is invalid."
"Rerun with integer scaling factors.\n");
return;
}
#endif
for (int k = num_tile_rows - 1; k >= 0; k--) {
for (int l = 0; l < num_tile_cols; l++) {
const int num_rows = first_row[k + 1] - first_row[k];
const int num_cols = first_col[l + 1] - first_col[l];
scaling_t ratio;
#ifdef INTSCALING
ratio = *scale - scales(k,l);
#else
ratio = *scale / scales(k,l);
#endif
if (mem_layout == COLUMN_MAJOR)
scale_tile(num_rows, num_cols, C_tiles[k][l], ldC, &ratio);
else 
scale_tile(num_rows, num_cols, C_tiles[k][l], num_rows, &ratio);
}
}
destroy_2D_lock_grid(num_tile_rows, num_tile_cols, &lock[0][0]);
free(A_norms);
free(B_norms);
free(C_norms);
free(scales);
}
