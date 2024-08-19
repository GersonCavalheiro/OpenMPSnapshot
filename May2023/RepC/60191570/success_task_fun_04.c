#include <stdio.h>
#include <stdlib.h>
#pragma omp target device(smp) copy_in(*a) copy_out(*b)
#pragma omp task
void f(int *a, int *b)
{
*b = *a + 1;
}
int main(int argc, char *argv[])
{
int t_b = 1;
int t_a = 10;
f(&t_a, &t_b);
#pragma omp taskwait
if (t_b != 11)
{
fprintf(stderr, "t_b == %d != 11\n", t_b);
abort();
}
return 0;
}
