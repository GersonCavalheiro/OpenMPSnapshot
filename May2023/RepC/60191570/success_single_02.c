#include <stdlib.h>
int main(int argc, char *argv[])
{
int x, y, z;
x = 1;
y = 42;
z = 1;
#pragma omp single private(x) firstprivate(y)
{
x = 99;
if (y != 42)
{
abort();
}
y = 99;
z = 99;
}
if (x != 1)
{
abort();
}
if (y != 42)
{
abort();
}
if (z != 99)
{
abort();
}
return 0;
}
