#include <stdlib.h>
int main(int argc, char* argv[])
{
int k = 1;
#pragma omp parallel private(k)
{
k = 2;
#pragma omp task firstprivate(k)
{
if (k != 2)
abort();
k = 3;
}
#pragma omp taskwait
if (k != 2)
abort();
}
if (k != 1)
abort();
return 0;
}
