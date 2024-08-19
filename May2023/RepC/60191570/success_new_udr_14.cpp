#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
struct A
{
};
#pragma omp declare reduction (myop: A : omp_out=omp_in)
struct B: A
{
};
int main(int argc, char* argv[])
{
A b;
int r;
#pragma omp parallel reduction(myop: b)
r = rand();
printf("The random result is '%d'", r);
return 0;
}
