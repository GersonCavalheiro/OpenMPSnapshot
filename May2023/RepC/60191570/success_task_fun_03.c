#include <stdio.h>
#include <stdlib.h>
#pragma omp target device (smp)
#pragma omp task inout(*a)
void task1(int *a)
{
*a += 2;
}
#pragma omp target device (smp) implements(task1)
#pragma omp task inout(*a)
void task1_smp_v2(int *a)
{
*a += 2;
}
int main(int argc, char *argv[])
{
int a = 5;
task1(&a);
#pragma omp taskwait
if (a != 7)
{
fprintf(stderr, "a == %d != 7\n", a);
abort();
}
return 0;
}
