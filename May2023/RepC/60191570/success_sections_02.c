#include <stdlib.h>
#include <stdio.h>
int main (int argc, char *argv[])
{
int a = 0, b = 0;
#pragma omp parallel
#pragma omp sections
{
#pragma omp section
a = 1;
#pragma omp section
b = 2;
}
if (a == 0 || b == 0)
{
fprintf(stderr, "a = %d | b = %d\n", a, b);
abort();
}
return 0;
}
