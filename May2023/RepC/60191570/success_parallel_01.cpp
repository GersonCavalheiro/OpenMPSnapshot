#include <stdlib.h>
#include "omp.h"
int main(int argc, char* argv[])
{
int a = 3;
#pragma omp parallel if (a > 50)
{
if (omp_get_thread_num() != 0)
abort();
}
}
