#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 10
int main() {
int compute_array[OMPVV_NUM_THREADS_HOST][N];
int errors = 0;
int i, j;
int real_num_threads;
int is_offloading;
OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);
for (i = 0; i < OMPVV_NUM_THREADS_HOST; i++) {
for (j = 0; j < N; j++) {
compute_array[i][j] = 0;
}
}
omp_set_num_threads(OMPVV_NUM_THREADS_HOST);
#pragma omp parallel
{
int fp_val = omp_get_thread_num();
int p_val=0;
real_num_threads = omp_get_num_threads();
#pragma omp target map(tofrom:compute_array[fp_val][0:N]) firstprivate(fp_val) private(p_val)
{
p_val = fp_val;
for (i = 0; i < N; i++)
compute_array[p_val][i] += p_val;
} 
}
for (i = 0; i < real_num_threads; i++) {
for (j = 0; j < N; j++){
OMPVV_TEST_AND_SET_VERBOSE(errors, (compute_array[i][j] != i));
}
}
OMPVV_REPORT_AND_RETURN(errors);
}
