#include <stdlib.h>
namespace A {
#pragma omp declare reduction(myop: int: omp_out = omp_in + omp_out)
}
int main(int argc, char* argv[])
{
int x;
#pragma omp parallel reduction(myop : x)
x = rand();
return 0;
}
