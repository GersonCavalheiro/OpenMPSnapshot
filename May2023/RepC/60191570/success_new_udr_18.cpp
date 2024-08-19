#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#pragma omp declare reduction (add: int: omp_out=omp_in+omp_out)
int main (int argc, char* argv[])
{
int a = 0;
#pragma omp parallel firstprivate(a) reduction (add: a)
a = a + 5;
printf("reduction done");
return 0;
}
