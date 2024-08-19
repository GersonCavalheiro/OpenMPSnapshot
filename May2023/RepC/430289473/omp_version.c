#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "coo2csc.h"
#include "mergesort.h"
#include "mmio.h"
#include "utils.h"
int M, N, nz;
int *lower_coo_rows, *lower_coo_cols, *lower_coo_values;
int *indices;
int *pointers;
int *values;
int *lower_res_rows, *lower_res_cols, *lower_res_values;
int *res_indices, *res_pointers, *res_values;
int res_nnz = 0;
omp_lock_t lock;
int main(int argc, char *argv[]) {
double t = clock();
char *filepath;
MM_typecode *matcode;
int ret_code;
int num_threads;
parse_cli_args_parallel(argc, argv, &filepath, &num_threads);
omp_set_dynamic(0);
omp_set_num_threads(num_threads);
ret_code = mm_read_mtx_crd(filepath, &M, &N, &nz, &lower_coo_rows,
&lower_coo_cols, &lower_coo_values, &matcode);
if (ret_code) {
fprintf(stderr, "Error while reading mm file.");
exit(1);
}
for (int i = 0; i < nz; i++) {
lower_coo_rows[i]--;
lower_coo_cols[i]--;
}
int *full_rows = (int *)malloc(2 * nz * sizeof(int));
int *full_cols = (int *)malloc(2 * nz * sizeof(int));
int *full_values = (int *)malloc(2 * nz * sizeof(int));
for (int i = 0; i < nz; i++) {
full_rows[i] = lower_coo_rows[i];
full_rows[nz + i] = lower_coo_cols[i];
full_cols[i] = lower_coo_cols[i];
full_cols[nz + i] = lower_coo_rows[i];
}
indices = (int *)malloc(2 * nz * sizeof(int));
pointers = (int *)malloc((N + 1) * sizeof(int));
coo2csc(indices, pointers, values, full_rows, full_cols, full_values, 2 * nz,
N, 0, 1);
for (int i = 0; i < N; i++) {
mergeSort(indices, pointers[i], pointers[i + 1] - 1);
}
lower_res_rows = (int *)malloc(nz * sizeof(int));
lower_res_cols = (int *)malloc(nz * sizeof(int));
lower_res_values = (int *)malloc(nz * sizeof(int));
int result;
omp_init_lock(&lock);
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < nz; ++i) {
result = get_common_subarray_items_count(
indices, pointers[lower_coo_rows[i]], pointers[lower_coo_rows[i] + 1],
pointers[lower_coo_cols[i]], pointers[lower_coo_cols[i] + 1]);
if (result != 0) {
omp_set_lock(&lock);
lower_res_rows[res_nnz] = lower_coo_rows[i];
lower_res_cols[res_nnz] = lower_coo_cols[i];
lower_res_values[res_nnz] = result;
res_nnz++;
omp_unset_lock(&lock);
}
}
}
lower_res_rows = (int *)realloc(lower_res_rows, res_nnz * sizeof(int));
lower_res_cols = (int *)realloc(lower_res_cols, res_nnz * sizeof(int));
lower_res_values = (int *)realloc(lower_res_values, res_nnz * sizeof(int));
int *full_res_rows = (int *)malloc(2 * res_nnz * sizeof(int));
int *full_res_cols = (int *)malloc(2 * res_nnz * sizeof(int));
int *full_res_values = (int *)malloc(2 * res_nnz * sizeof(int));
for (int i = 0; i < res_nnz; i++) {
full_res_rows[i] = lower_res_rows[i];
full_res_rows[res_nnz + i] = lower_res_cols[i];
full_res_cols[i] = lower_res_cols[i];
full_res_cols[res_nnz + i] = lower_res_rows[i];
full_res_values[i] = lower_res_values[i];
full_res_values[res_nnz + i] = lower_res_values[i];
}
res_indices = (int *)malloc(2 * res_nnz * sizeof(int));
res_pointers = (int *)malloc((N + 1) * sizeof(int));
res_values = (int *)malloc(2 * res_nnz * sizeof(int));
coo2csc(res_indices, res_pointers, res_values, full_res_rows, full_res_cols,
full_res_values, 2 * res_nnz, N, 0, 0);
for (int i = 0; i < N; i++) {
mergeSort_mirror(res_indices, res_values, res_pointers[i],
res_pointers[i + 1] - 1);
}
int *triangles_vector = (int *)calloc(N, sizeof(int));
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < N; i++) {
for (int j = res_pointers[i]; j < res_pointers[i + 1]; j++) {
triangles_vector[i] += res_values[j];
}
#pragma omp atomic
triangles_vector[i] /= 2;
}
}
int triangle_count = 0;
for (int i = 0; i < N; i++) {
triangle_count += triangles_vector[i];
}
double total_time = (double)(clock() - t) / CLOCKS_PER_SEC;
printf("Triangle count : %d\n", triangle_count / 3);
fprintf(stdout, "Total execution time: %lf\n", total_time);
}