#include <stdio.h>
#include <omp.h>

int main()
{
int x = 0;

#pragma omp parallel shared(x)
{
x = 42;
}
printf("X: %d\n", x);
}
