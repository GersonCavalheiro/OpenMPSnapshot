#include <stdlib.h>
#include <stdio.h>
int a;
int main(int argc, char *argv[])
{
int b;
int i;
a = 3;
b = 4;
#pragma omp for private(a, b)
for (i = 0; i < 10; i++)
{
a = a + 3;
b = b + 4;
}
if (a != 3)
{
fprintf(stderr, "%d != 3\n", a);
abort();
}
if (b != 4)
{
fprintf(stderr, "%d != 4\n", b);
abort();
}
return 0;
}
