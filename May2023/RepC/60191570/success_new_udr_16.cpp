#include "omp.h"
namespace n
{
struct A {};
#pragma omp declare reduction (foo: A: omp_out=omp_in)
}
int main (int argc, char* argv[])
{
n::A a;
#pragma omp parallel reduction (foo: a)
a;
return 0;
}
