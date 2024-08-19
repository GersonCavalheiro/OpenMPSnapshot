#include <stdio.h>
#include <omp.h>
int main()
{
int i = 1;
int j = 2;
printf("\nInitial declaration: i=%d j=%d", i, j);
#pragma omp parallel private(i) firstprivate(j)
{
printf("\nInside construct 1: i=%d j=%d", i, j);
i = 3;
j = j + 2;
printf("\nInside construct 2: i=%d j=%d", i, j);
}
printf("\n\nOutside construct: i=%d j=%d", i, j);
return 0;
}
