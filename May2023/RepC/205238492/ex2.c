#include <stdio.h>
#include <omp.h>
int main(void)
{
int threadId;
#pragma omp parallel
{
threadId = omp_get_thread_num();
printf("\nOi %d\n", threadId);
}
return 0;
}
