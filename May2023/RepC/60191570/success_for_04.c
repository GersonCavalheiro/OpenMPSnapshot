#include <stdlib.h>
int a;
int main(int argc, char *argv[])
{
int b;
int i;
a = 3;
b = 4;
#pragma omp for firstprivate(a, b)
for (i = 0; i < 10; i++)
{
if (a != (3 + i))
abort();
if (b != (4 + i))
abort();
a++;
b++;
}
if (a != 3)
abort();
if (b != 4)
abort();
return 0;
}
