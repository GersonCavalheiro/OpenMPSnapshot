#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>
int main(int argc, char *argv[])
{
int id;
int proc_num;
int thread_num;
double wtime;
double wtime1;
double wtime2;
printf("\n");
printf("HELLO\n");
printf("  C/OpenMP version\n");
wtime1 = omp_get_wtime();
proc_num = omp_get_num_procs();
printf("\n");
printf("  The number of processors available:\n");
printf("  OMP_GET_NUM_PROCS () = %d\n", proc_num);
#pragma omp parallel private(id)
{
id = omp_get_thread_num();
thread_num = omp_get_num_threads();
if (id == 0)
{
printf("\n");
printf("  Calling OMP_GET_NUM_THREADS inside a\n");
printf("  parallel region, we get the number of\n");
printf("  threads is %d\n", thread_num);
}
printf("  This is process %d out of %d\n", id, thread_num);
}
thread_num = 2 * thread_num;
printf("\n");
printf("  We request %d threads.\n", thread_num);
omp_set_num_threads(thread_num);
#pragma omp parallel private(id)
{
id = omp_get_thread_num();
thread_num = omp_get_num_threads();
if (id == 0)
{
printf("\n");
printf("  Calling OMP_GET_NUM_THREADS inside a\n");
printf("  parallel region, we get the number of\n");
printf("  threads is %d\n", thread_num);
}
printf("  This is process %d out of %d\n", id, thread_num);
}
wtime2 = omp_get_wtime();
wtime = wtime2 - wtime1;
printf("\n");
printf("HELLO\n");
printf("  Normal end of execution.\n");
printf("\n");
printf("  Elapsed wall clock time = %f\n", wtime);
return 0;
}
