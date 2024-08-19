#include <stdlib.h>
void g(int **a)
{
int ** b = a;
#pragma omp task inout([10](a[3])) firstprivate(b) no_copy_deps
{
if (a != b)
{
abort();
}
}
#pragma omp taskwait
}
int main(int argc, char* argv[])
{
int a0[2] = {0,1};
int a1[2] = {2,3};
int *a[2] = { a0, a1 };
g(a);
return 0;
}
