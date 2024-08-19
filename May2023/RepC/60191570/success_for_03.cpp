#include <stdlib.h>
int main(int arch, char *argv[])
{
int c[30];
int x = 10, y = 20, i;
#pragma omp for
for (i=0; i < x + y; i = i + 2)
{
c[i] = i;
}
for (i=0; i < x + y; i = i + 2)
{
if (c[i] != i) abort();
}
#pragma omp parallel for
for (i=0; i < x + y; i = i + 2)
{
c[i] = 2*i;
}
for (i=0; i < x + y; i = i + 2)
{
if (c[i] != 2*i) abort();
}
return 0;
}
