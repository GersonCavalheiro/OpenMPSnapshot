#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
void hello(void)
{
int my_rank = omp_get_thread_num();
int thread_count = omp_get_num_threads();
printf("--Hello--");
printf("from thread %d of %d\n", my_rank, thread_count);
}
int main(int argc, char *argv[])
{
int thread_count = 4;
if (argc > 1)
thread_count = atoi(argv[1]);
int p = omp_get_num_procs();
printf("p = %d\n", p);
#pragma omp parallel num_threads(thread_count)
{
hello();
}
printf("done\n");
return 0;
}
