#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
char a1 = 0;
char a2 = 0;
#pragma omp sections shared(a1, a2)
{
#pragma omp section
{
a1 = 1;
}
#pragma omp section
{
a2 = 1;
}
}
if (!a1 || !a2)
abort();
return 0;
}
