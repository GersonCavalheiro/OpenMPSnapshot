#include "majorant.h"
#include "norm.h"
#include <stdio.h>
void bound_triangular_matrix(
double ***A_tiles, int ldA,
double *restrict A_norms,
int num_tiles, int *p,
matrix_desc_t type,
memory_layout_t mem_layout)
{
#define A_norms(i,j) A_norms[(i) + (j) * num_tiles]
switch(type) {
case UPPER_TRIANGULAR:
{
for (int i = 0; i < num_tiles; i++) {
#pragma omp task
for (int j = i; j < num_tiles; j++) {
const int m = p[i + 1] - p[i];
const int n = p[j + 1] - p[j];
if (m > 0 && n > 0) {
if (mem_layout == COLUMN_MAJOR)
A_norms(i,j) = matrix_infnorm(m, n, A_tiles[i][j], ldA);
else 
A_norms(i,j) = matrix_infnorm(m, n, A_tiles[i][j], m);
}
}
}
break;
}
case LOWER_TRIANGULAR:
{
for (int i = 0; i < num_tiles; i++) {
#pragma omp task
for (int j = 0; j <= i; j++) {
const int m = p[i + 1] - p[i];
const int n = p[j + 1] - p[j];
if (mem_layout == COLUMN_MAJOR)
A_norms(i,j) = matrix_infnorm(m, n, A_tiles[i][j], ldA);
else 
A_norms(i,j) = matrix_infnorm(m, n, A_tiles[i][j], m);
}
}
break;
}
default:
{
break;
}
}
#undef A_norms
}
void compute_column_majorants(
int m, int n,
const double *restrict C, int ldC,
double *restrict C_norms)
{
for (int j = 0; j < n; j++) {
C_norms[j] = vector_infnorm(m, C + j * ldC);
}
}
