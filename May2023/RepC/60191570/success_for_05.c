#include <stdlib.h>
#include <stdio.h>
int a;
int main(int argc, char *argv[])
{
int b;
int i;
a = 3;
b = 4;
#pragma omp parallel for lastprivate(a, b)
for (i = 0; i < 10; i++)
{
a = i;
b = i + 1;
}
if (a != 9)
{
fprintf(stderr, "a == %d != %d\n", a, 9);
abort();
}
if (b != 10)
{
fprintf(stderr, "b == %d != %d\n", b, 10);
abort();
}
return 0;
}
