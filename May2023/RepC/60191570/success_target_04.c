#include <stdlib.h>
#include <stdio.h>
int my_global = 1;
int main(int argc, char *argv[])
{
int cpinout = 9;
#pragma omp target device(smp) copy_inout(my_global, cpinout)
#pragma omp task 
{
my_global++;
cpinout++;
}
#pragma omp taskwait
if (my_global != 2)
{
fprintf(stderr, "my_global == %d != 2\n", my_global);
abort();
}
if (cpinout != 10)
{
fprintf(stderr, "cpinout == %d != 10\n", cpinout);
abort();
}
return 0;
}
