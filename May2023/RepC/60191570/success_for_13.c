#include <stdlib.h>
int main(int argc, char* argv[])
{
int x, k = 3;
#pragma omp parallel private(x)
{
#pragma omp for
for (x = 0; x < 10; x++)
{
k = 12;
}
}
if (k != 12)
abort();
return 0;
}
