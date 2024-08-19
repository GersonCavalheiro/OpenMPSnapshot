#include <stdlib.h>
#include "omp.h"
struct A
{
};
#pragma omp declare reduction (foo: A : omp_out=omp_in)
struct B:A {};
struct C:A {};
struct D:B,C {
};
int main (int argc, char* argv[])
{
D d;
int x;
#pragma omp parallel reduction (foo : d)
x = rand();
return 0;
}
