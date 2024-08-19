#include <stdlib.h>
int a;
#pragma omp threadprivate(a)
int main(int argc, char *argv[])
{
#pragma omp parallel
{
a = omp_get_thread_num();
}
#pragma omp parallel
{
if (a != omp_get_thread_num())
abort();
}
return 0;
}
