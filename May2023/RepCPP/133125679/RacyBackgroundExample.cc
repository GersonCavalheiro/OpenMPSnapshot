
#include <stdio.h>

#if !defined(NTHREADS)
#define NTHREADS 2
#endif

int main() {
int i=0;
#pragma omp parallel num_threads(NTHREADS)
#pragma omp single
{
#pragma omp task shared(i)
#pragma omp critical(lock_i)
{  i = 1; }

#pragma omp task shared(i)
#pragma omp critical(lock_i)
{ i = 2; }
}

printf ("i=%d\n",i);
return 0;
}
