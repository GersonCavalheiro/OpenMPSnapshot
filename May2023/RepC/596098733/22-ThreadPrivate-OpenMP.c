#include <stdio.h>
#include <omp.h>
#include <stdio.h>
#include <omp.h>
static int private_var_outer;
#pragma omp threadprivate(private_var_outer)
int main() {
int num_threads, tid;
#pragma omp parallel num_threads(2)
{
#pragma omp single
{
num_threads = omp_get_num_threads();
printf("Number of threads = %d\n", num_threads);
}
tid = omp_get_thread_num();
if (tid == 0) {
private_var_outer = 0;
printf("Thread %d: private_var_outer = %d\n", tid, private_var_outer);
} else {
private_var_outer = 1;
printf("Thread %d: private_var_outer = %d\n", tid, private_var_outer);
}
#pragma omp barrier
#pragma omp master
{
printf("Master thread (%d) resetting private_var_outer\n", tid);
private_var_outer = 42;
}
#pragma omp barrier
printf("Thread %d: private_var_outer = %d\n", tid, private_var_outer);
#pragma omp barrier
#pragma omp parallel num_threads(2)
{
static int private_var_inner;
#pragma omp threadprivate(private_var_inner)
int tid_inner = omp_get_thread_num();
if (tid_inner == 0) {
private_var_inner = 0;
printf("Thread %d: private_var_inner = %d\n", tid, private_var_inner);
} else {
private_var_inner = 1;
printf("Thread %d: private_var_inner = %d\n", tid, private_var_inner);
}
#pragma omp barrier
#pragma omp master
{
printf("Master thread (%d) resetting private_var_inner\n", tid_inner);
private_var_inner = 42;
}
#pragma omp barrier
printf("Thread %d: private_var_inner = %d\n", tid, private_var_inner);
#pragma omp barrier
}
}
return 0;
}
