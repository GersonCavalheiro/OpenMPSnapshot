#include <stdlib.h>
#include "omp.h"
struct A
{
static int x;
#pragma omp threadprivate(x)
};
int A::x = 9;
int main(int argc, char *argv[])
{
if (A::x != 9)
abort();
#pragma omp parallel default(none) 
{
if (A::x != 9)
abort();
A::x = omp_get_thread_num();
}
#pragma omp parallel default(none) 
{
if (A::x != omp_get_thread_num())
abort();
}
return 0;
}
