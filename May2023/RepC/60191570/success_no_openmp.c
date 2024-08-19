#include <stdlib.h>
int main(int argc, char *argv[])
{
int x = 0;
#pragma omp parallel
{
#pragma omp atomic
x++;
}
if (x != 1)
abort();
return 0;
}
