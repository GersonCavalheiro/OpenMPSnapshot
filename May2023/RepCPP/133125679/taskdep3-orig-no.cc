


#include <stdio.h>

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
printf ("x=%d\n", i);
#pragma omp task depend (in:i)
printf ("x=%d\n", i);
}

return 0;
}
