


#include <assert.h>

#if !defined(NTHREADS)
#define NTHREADS 2
#endif

int main()
{
int i=0;
#pragma omp parallel num_threads(NTHREADS)
#pragma omp single
{
#pragma omp task depend (out:i)
i = 1;
#pragma omp task depend (in:i)
i = 2;
}

assert (i==2);
return 0;
}
