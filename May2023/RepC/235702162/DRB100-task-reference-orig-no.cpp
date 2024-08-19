#if (_OPENMP<201511)
#error "OpenMP 4.5 compilers (e.g. GCC 6.x or later ) are needed to compile this test."
#endif
#include <stdio.h>
#define MYLEN 100
int a[MYLEN];
void gen_task(int& i)
{
#pragma omp task
{
a[i]= i+1;
}
}
int main()
{
int i=0;
#pragma omp parallel
{
#pragma omp single
{
for (i=0; i<MYLEN; i++)
{
gen_task(i);
}
}
}
for (i=0; i<MYLEN; i++)
{
if (a[i]!= i+1)
{
printf("warning: a[%d] = %d, not expected %d\n", i, a[i], i+1);
}
}
return 0;
}
