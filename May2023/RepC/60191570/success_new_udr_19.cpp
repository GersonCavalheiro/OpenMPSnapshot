#include <stdlib.h>
#pragma omp declare reduction(myop: int: omp_out = omp_in * omp_out)
namespace A {
int foo;
#pragma omp declare reduction(myop: int: omp_out = omp_in + omp_out)
}
int main(int argc, char* argv[])
{
int x;
using namespace A;
#pragma omp parallel reduction(A::myop : x)
x = rand();
return 0;
}
