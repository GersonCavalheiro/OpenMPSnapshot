#include <stdlib.h>
#include "omp.h"
int main(int argc, char* argv[])
{
int people = 0;
#pragma omp parallel
{
#pragma omp master
{
if (omp_get_thread_num() != 0)
abort();
people++;
}
#pragma omp barrier
}
if (people != 1)
abort();
return 0;
}
