#include <stdlib.h>
#include <stdio.h>
int a;
int b;
int main(int argc, char *argv[])
{
int i;
a = -3;
b = -4;
#pragma omp parallel for firstprivate(a, b) lastprivate(a, b)
for (i = 0; i < 100; i++)
{
if (a < 0) if (a != -3) abort();
if (b < 0) if (b != -4) abort();
a = i;
b = i;
}
if (a != 99)
abort();
if (b != 99)
abort();
return 0;
}
