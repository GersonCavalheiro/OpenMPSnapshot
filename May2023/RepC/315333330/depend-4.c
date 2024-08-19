#include <stdlib.h>
#include <unistd.h>
int
main ()
{
#pragma omp parallel
#pragma omp single
{
int x = 1, y = 2, z = 3;
#pragma omp taskgroup
{
#pragma omp task shared (x, y, z) depend(inout: x, y) depend (in: z) if (x > 10)
{
if (x != 1 || y != 2 || z != 3)
abort ();
x = 4;
y = 5;
}
if (x != 4 || y != 5)
abort ();
}
#pragma omp taskgroup
{
#pragma omp task shared (x, y) depend(in: x, y)
{
usleep (10000);
if (x != 4 || y != 5 || z != 3)
abort ();
}
#pragma omp task shared (x, y) depend(in: x, y)
{
usleep (10000);
if (x != 4 || y != 5 || z != 3)
abort ();
}
#pragma omp task shared (x, y, z) depend(inout: x, y) depend (in: z) if (x > 10)
{
if (x != 4 || y != 5 || z != 3)
abort ();
x = 6;
y = 7;
}
}
if (x != 6 || y != 7)
abort ();
}
return 0;
}
