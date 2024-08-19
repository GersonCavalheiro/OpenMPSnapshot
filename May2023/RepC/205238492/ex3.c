#include <stdio.h>
#include <omp.h>
int main(void)
{
int threadId, nThreads;
#pragma omp parallel private(threadId, nThreads)
{
threadId = omp_get_thread_num();
printf("\nOi %d\n", threadId);
if (threadId == 0) {
nThreads = omp_get_num_threads();
printf("QTD threads = %d\n", nThreads);
}
}
return 0;
}
