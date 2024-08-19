#include <math.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"

int main(int argc, const char * argv[])
{
const int size = 10;
double sinTable[size];
memset(sinTable,0,sizeof(sinTable));
auto before1 = omp_get_wtime();

#pragma omp parallel
{
#pragma omp single
{
printf("single %d\n",omp_get_thread_num());
for(int n=0; n<size; ++n)
{
#pragma omp task default(none) shared(sinTable) firstprivate(n) 
{
#pragma omp critical
{
printf("thread %d iter %d %p\n", omp_get_thread_num() ,n,sinTable);
}
sinTable[n] = sin(2 * M_PI * n / size);
}
}
#pragma omp taskwait
}

}
auto after1 = omp_get_wtime();

printf("elapsed omp(s): %f\n" ,(double) (after1-before1) );
for(int n=0; n<size; ++n)
{
#if 0
printf("result %d %f\n",n,sinTable[n]);
#endif
}
return 0;

}