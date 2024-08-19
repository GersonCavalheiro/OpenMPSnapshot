#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAX_ELEMS 100
int main(int argc, char *argv[])
{
int c[100 + MAX_ELEMS + 100];
int i;
memset(c, 0, sizeof(c));
#pragma omp for
for (i = MAX_ELEMS; i > 0; i--)
{
c[i + 100 - 1] = i - 1;
}
for (i = 0; i < MAX_ELEMS; i++)
{
if (c[i + 100] != i)
{
fprintf(stderr, "c[%d] == %d != %d\n", i + 100, c[i + 100], i);
abort();
}
}
for (i = 0; i < 100; i++)
{
if (c[i] != 0)
{
fprintf(stderr, "c[%d] == %d != %d\n", i, c[i], 0);
abort();
}
if (c[100 + MAX_ELEMS + i] != 0)
{
fprintf(stderr, "c[%d] == %d != %d\n", 100 + MAX_ELEMS + i, c[100 + MAX_ELEMS + i], 0);
abort();
}
}
return 0;
}
