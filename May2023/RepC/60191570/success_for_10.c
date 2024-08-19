#include <stdlib.h>
int a[10];
int main(int argc, char *argv[])
{
int b[10];
int i;
a[1] = 3;
a[2] = 2;
b[4] = 4;
b[5] = 5;
#pragma omp parallel for firstprivate(a, b) lastprivate(a, b)
for (i = 0; i < 10; i++)
{
a[1] = a[2];
b[4] = b[5];
}
if (a[1] != 2)
abort();
if (a[2] != 2)
abort();
if (b[4] != 5)
abort();
if (b[5] != 5)
abort();
return 0;
}
