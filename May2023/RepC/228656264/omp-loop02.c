#include <omp.h>
extern void abort (void);
#define N 10
void parloop (int *a)
{
int i;
#pragma omp for
for (i = 0; i < N; i++)
a[i] = i + 3;
}
int
main()
{
int i, a[N];
#pragma omp parallel shared(a)
{
parloop (a);
}
for (i = 0; i < N; i++)
if (a[i] != i + 3)
abort ();
return 0;
}
