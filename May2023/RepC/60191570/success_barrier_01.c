#include <stdlib.h>
int main(int argc, char *argv[])
{
int i = 0;
int *p = &i;
#pragma omp parallel shared(i) firstprivate(p)
{
if (p != &i)
abort();
int j;
for (j = 0; j < 100; j++)
{
int k;
for (k = 0; k < 100; k++)
{
if (i != 0)
abort();
}
}
#pragma omp barrier
i = 1;
}
return 0;
}
