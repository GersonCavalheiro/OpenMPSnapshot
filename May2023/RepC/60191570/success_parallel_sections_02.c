#include <stdlib.h>
int main(int argc, char *argv[])
{
int s = 0;
#pragma omp parallel sections reduction(+:s)
{
#pragma omp section
{
s = s + 1;
}
#pragma omp section
{
s = s + 1;
}
#pragma omp section
{
s = s + 1;
}
}
if (s != 3)
abort();
return 0;
}
