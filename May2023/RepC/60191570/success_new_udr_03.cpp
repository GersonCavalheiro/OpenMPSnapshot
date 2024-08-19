#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
namespace A {
#pragma omp declare reduction(myop: int: omp_out = omp_in + omp_out)
int foo;
}
int main(int argc, char* argv[])
{
int x;
using namespace A;
#pragma omp parallel reduction(myop : x)
x = rand();
printf("The random result is '%d'", x);
return 0;
}
