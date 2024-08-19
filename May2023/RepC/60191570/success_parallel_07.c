#include <stdlib.h>
int a[10];
int main(int argc, char *argv[])
{
int b[10];
a[2] = 3;
b[5] = 4;
#pragma omp parallel firstprivate(a, b)
{
int i;
for (i = 0; i < 10; i++)
{
if (a[2] != (3 + i))
abort();
if (b[5] != (4 + i))
abort();
a[2]++;
b[5]++;
}
}
if (a[2] != 3)
abort();
if (b[5] != 4)
abort();
return 0;
}
