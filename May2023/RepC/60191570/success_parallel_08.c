#include <stdlib.h>
int main(int argc, char *argv[])
{
int a = 2;
#pragma omp parallel if (a > 50)
{
if (omp_get_thread_num() != 0)
{
abort();
}
}
return 0;
}
