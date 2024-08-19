#include <stdlib.h>
int a;
int main(int argc, char *argv[])
{
int b;
a = 10;
b = 20;
#pragma omp parallel firstprivate(a, b)
{
int i;
for (i = 0; i < 10; i++)
{
if (a != (10 + i))
abort();
if (b != (20 + i))
abort();
a++;
b++;
}
}
if (a != 10)
abort();
if (b != 20)
abort();
return 0;
}
