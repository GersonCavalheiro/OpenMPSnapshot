#include <stdlib.h>
int a[2];
#pragma omp threadprivate(a)
int main(int argc, char *argv[])
{
#pragma omp parallel
{
a[1] = omp_get_thread_num();
}
#pragma omp parallel
{
if (a[1] != omp_get_thread_num())
abort();
}
return 0;
}
