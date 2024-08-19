#include "p21fun.h"
int main(int argc, char * argv[]) {
argcheck(argc, 3);
printf("[Core master]:\n"
"file: %s\n"
"np: %s\n\n",
argv[1], argv[2]);
int np = (int) strtol(argv[2], NULL, 10);
if (np < 1) easy_stderr("np should be at least 1!", -1);
printf("npxnp: %d\n\n", np*np);
struct matrix * matrix = matrix_alloc(matrix_init(argv[1]));
printf("matrix of size %dx%d:\n", matrix->M, matrix->M);
matrix_square_print(matrix->mtx_ptr, matrix->M);
printf("\n");
printf("[MASTER THREAD]:\n"
"RECAP:\n"
"np: %d\n"
"npxnp: %d\n"
"M: %d\n"
"block_size: %d\n"
"block_no: %d\n\n",
np, np*np, matrix->M, matrix->M/np, np*np);
if (np > matrix->M) easy_stderr("np must be <= M!", -1);
omp_set_num_threads(np*np);
struct matrix * mtx_array = (struct matrix *) malloc(sizeof(struct matrix) * (np*np));
int tid;
#pragma omp parallel default(none), shared(matrix, np, mtx_array), private(tid)
{
tid = omp_get_thread_num();
mtx_array[tid] = * matrix_block_extract(matrix, np, tid);
}
for (int i = 0; i < np*np; ++i) {
printf("[Block %d]:\n", i);
matrix_square_print(mtx_array[i].mtx_ptr, mtx_array[0].M);
printf("\n");
}
omp_set_num_threads(np);
struct matrix * sum_matrix = initialize_sum_matrix(matrix->M/np);
struct matrix * temp_sum;
int nloc;
int actual_threads_no;
int reminder;
int step;
double parallel_init_time = omp_get_wtime();
#pragma omp parallel default(none), shared(np, mtx_array, sum_matrix, reminder), private(tid, nloc, temp_sum, actual_threads_no, step)
{
actual_threads_no = omp_get_num_threads();
tid = omp_get_thread_num();
nloc = (np*np) / actual_threads_no;
reminder = (np*np) % actual_threads_no;
temp_sum = initialize_sum_matrix(sum_matrix->M);
if (tid < reminder) {
nloc++;
step = 0;
}
else {
step = reminder;
}
for (int i = 0; i < nloc; ++i) {
temp_sum = matrix_sum(temp_sum, &mtx_array[i + nloc * tid + step]);
}
#pragma omp critical
sum_matrix = matrix_sum(sum_matrix, temp_sum);
}
double parallel_end_time = omp_get_wtime();
printf("[Core %d]: PARALLEL Sum matrix: \n", omp_get_thread_num());
matrix_square_print(sum_matrix->mtx_ptr, sum_matrix->M);
printf("\n");
double sequential_init_time = omp_get_wtime();
struct matrix * sequential_sum = initialize_sum_matrix(sum_matrix->M);
for (int i = 0; i < np*np; ++i) {
sequential_sum = matrix_sum(sequential_sum, &mtx_array[i]);
}
double sequential_end_time = omp_get_wtime();
printf("[Core %d]: SEQUENTIAL Sum matrix: \n", omp_get_thread_num());
matrix_square_print(sequential_sum->mtx_ptr, sequential_sum->M);
printf("\n");
if (matrix_compare(sequential_sum, sum_matrix) == -1) {
easy_stderr("Sum matrix is not what it should be!", -1);
}
double parallel_exec_time = parallel_end_time - parallel_init_time;
double sequential_exec_time = sequential_end_time - sequential_init_time;
printf("[EXECUTION TIMES]:\n"
"Parallel algorithm with np = %d: [%f]\n"
"Sequential algorithm: [%f]", np, parallel_exec_time, sequential_exec_time);
return 0;
}
